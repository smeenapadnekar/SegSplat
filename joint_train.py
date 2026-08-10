#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
from random import randint
from utils.loss_utils import l1_loss, l1_loss_mask, l2_loss, ssim, second_order_tv, smooth_3d_intra_class, protoCL, smooth_3d_inter_class
from utils.depth_utils import estimate_depth
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
from utils.load_backbone import FeatureExtractor
from lerf.lerf import LERFModel
from utils.lerf_optimizer_scedulers import ExponentialDecaySchedulerConfig,  ExponentialDecayScheduler
import torch.nn.functional as F
import random
from PIL import Image
import cv2
import torchvision.transforms as transforms

from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler, MultiControlNetModel, AutoencoderKL
from diffusers.utils import load_image, logging
logging.set_verbosity_error()
os.environ["DIFFUSERS_PROGRESS_BAR"] = "false"
seed = 55
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)



# for image generation 
# Load ControlNet for depth
depth_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",
    torch_dtype=torch.float16
)
# Load ControlNet for Canny Edge
canny_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)
multi_controlnet = MultiControlNetModel([depth_controlnet, canny_controlnet])
gen_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    # "stable-diffusion-v1-5/stable-diffusion-v1-5", 
    # "SG161222/Realistic_Vision_V5.1_noVAE", #good
    "lykon/dreamshaper-8",
    controlnet=multi_controlnet, 
    torch_dtype=torch.float16
).to("cuda")
gen_pipe.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
gen_pipe.scheduler = UniPCMultistepScheduler.from_config(gen_pipe.scheduler.config)
# gen_pipe.enable_model_cpu_offload()
# gen_pipe.enable_attention_slicing()
# gen_pipe.enable_vae_tiling()
gen_pipe.enable_xformers_memory_efficient_attention()
# Edge threshold 
low_threshold = 100
high_threshold = 200
transform_to_pil = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()

def sobel_filter(img: torch.Tensor, device="cuda") -> torch.Tensor:
    if img.shape[1] == 3:
        img = 0.2989 * img[:, 0:1] + 0.5870 * img[:, 1:2] + 0.1140 * img[:, 2:3]
    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    # Convolve
    grad_x = F.conv2d(img, sobel_x, padding=1)
    grad_y = F.conv2d(img, sobel_y, padding=1)

    # Gradient magnitude
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)  # avoid sqrt(0)

    # Normalize for visualization / importance map
    grad_magnitude = grad_magnitude / (grad_magnitude.max() + 1e-6)

    return grad_magnitude

def segmentation_boundary(seg: torch.Tensor,num_classes,device="cuda") -> torch.Tensor:
    one_hot = F.one_hot(seg, num_classes).permute(0, 3, 1, 2).float()

        # Define Sobel filters and expand to num_classes
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3).expand(num_classes, 1, 3, 3)

    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32, device=device).view(1, 1, 3, 3).expand(num_classes, 1, 3, 3)

    # Apply conv per channel (grouped conv)
    grad_x = F.conv2d(one_hot, sobel_x, padding=1, groups=num_classes)
    grad_y = F.conv2d(one_hot, sobel_y, padding=1, groups=num_classes)

    # Compute gradient magnitude and max over classes
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    grad_max = grad_mag.max(dim=1, keepdim=True).values  # (B, 1, H, W)

    # Optionally threshold
    boundary = (grad_max > 0.1).float()

    return boundary  # (B, 1, H, W)



def training(dataset, opt, pipe, args):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # MHE implementation from https://github.com/google-research/foundation-model-embedded-3dgs/blob/main/train.py
    lerf_model = None
    # lerf_featmap_manger = None
    # lerf_optimizer = None
    # lerf_opt_lr_scheduler = None
    # lerf_image_encoder = None
    b_vlrenderfeat = False

    if args.opt_vlrenderfeat_from > 0: # We initalize a lerfModel.
        b_vlrenderfeat = True
        lerf_model = LERFModel()
        lerf_model.cuda()
        lerf_model.train()
        print('Number of model parameters in Lerf: {}'.format(sum([p.data.nelement() for p in lerf_model.parameters()])))
        print('Number of optmizable model parameters in Lerf: {}'.format(sum([p.data.nelement() for p in lerf_model.parameters() if p.requires_grad])))

        lerf_optimizer = torch.optim.RAdam(lerf_model.parameters(), lr=args.fmap_lr, eps=1e-5, betas=(0.9, 0.999), weight_decay=1e-9)
        lerf_opt_scheduler_cfg = ExponentialDecaySchedulerConfig(lr_final=2e-3, max_steps=10000)

        LERF_LR_SCHEDULER = ExponentialDecayScheduler()
        lerf_opt_lr_scheduler = LERF_LR_SCHEDULER.get_scheduler(lerf_optimizer, lr_init=lerf_optimizer.param_groups[0]['lr'], config_=lerf_opt_scheduler_cfg)


    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    embedding_dim = args.embedding_dim
    #  Segmentation 
    num_classes = dataset.num_classes
    seg_decoder = torch.nn.Sequential(
        torch.nn.Conv2d(100, 1024, kernel_size=3, padding=1),  # 3x3 → padding=1
        torch.nn.ReLU(),
        torch.nn.Conv2d(1024, embedding_dim, kernel_size=3, padding=1),
        # torch.nn.ReLU(),
    )

    clip_head = torch.nn.Conv2d(embedding_dim, 100, kernel_size=3, padding=1)

    seg_head = torch.nn.Sequential(
        torch.nn.Conv2d(embedding_dim, num_classes, kernel_size=1, padding=0),
        # torch.nn.Softmax()  # make sure softmax is over class dimension
    )
    seg_decoder.cuda()
    clip_head.cuda()
    seg_head.cuda()
    cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    cls_decoder_optimizer = torch.optim.Adam(seg_decoder.parameters(),lr=5e-4)
    cls_head_optimizer = torch.optim.Adam(seg_head.parameters(),lr=5e-4)
    cls_clip_optimizer = torch.optim.Adam(clip_head.parameters(),lr=5e-4)
    seg_decoder.cuda()
    seg_head.cuda()
    viewpoint_stack = scene.getTrainCameras().copy()
    # for i in range(len(viewpoint_stack)):
    #     viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
    #     print(viewpoint_cam.uid)
    backbone_network_kd = FeatureExtractor(args.kd_backbone,embedding_dim,False,'cuda')
    cl_loss = protoCL(num_classes,100,temperature=0.2,queue_len=5)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    viewpoint_stack, pseudo_stack = None, None
    ema_loss_for_log = 0.0
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # Render
        bvl_feature_precomp = b_vlrenderfeat and (iteration > args.opt_vlrenderfeat_from - 1)
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            # num_viewpoints = len(viewpoint_stack)
            # viewpoints_opts = torch.arange(num_viewpoints).tolist()
            # viewpoint_stack = dict(zip(viewpoints_opts, viewpoint_stack))

        # print(num_viewpoints)

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # viewpoint_pop = randint(0, len(viewpoints_opts) - 1)
        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # viewpoint_indx = viewpoints_opts.pop(viewpoint_pop)
        # viewpoint_cam = viewpoint_stack.pop(viewpoint_indx) # .pop(viewpoint_indx) [viewpoint_indx]

        # GT
        gt_image = viewpoint_cam.original_image.cuda()
        gt_obj = viewpoint_cam.objects.cuda().long().clamp(0,num_classes-1)
        """
            High-frequency+Boundary map
        """

        importance_map = 0.8*sobel_filter(gt_image.unsqueeze(0))[0][0]+0.2*segmentation_boundary(gt_obj.unsqueeze(0),num_classes)[0][0]
        # importance_map = 1- importance_map
        # debug
        # arr = np.asarray(importance_map.cpu())
        # plt.imshow(arr)
        # plt.show()
        # print(a)

        render_pkg = render(
            viewpoint_cam, gaussians, pipe, background, 
            lerfmodel=lerf_model, 
            bvl_feature_precomp=bvl_feature_precomp, 
            fmap_resolution=args.fmap_resolution, 
            fmap_render_radiithre=args.fmap_render_radiithre,
            importance_map=importance_map,
            bg_map=1-importance_map,
        )

        # render_pkg_bg = render(
        #     viewpoint_cam, gaussians, pipe, background, 
        #     lerfmodel=lerf_model, 
        #     bvl_feature_precomp=bvl_feature_precomp, 
        #     fmap_resolution=args.fmap_resolution, 
        #     fmap_render_radiithre=args.fmap_render_radiithre,
        #     importance_map=1-importance_map
        # )
        image, viewspace_point_tensor, visibility_filter, radii, rendered_feature, segment_logit = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["rendered_featmap"], render_pkg["segment_logit"]
        
                                            # Cross Entropy
        seg_embedding_dec = seg_decoder(rendered_feature)
        # seg_embedding = clip_head(seg_embedding)
        seg_embedding = seg_head(seg_embedding_dec)     
        # seg_embedding =  F.log_softmax(seg_embedding,dim=1)
        # print("----",clip_embedding.shape,seg_embedding.shape)
        # gt_obj = F.one_hot(gt_obj, num_classes=200).permute( 2, 0, 1)
        # print(seg_embedding.shape,gt_obj.shape)                     
        # print(seg_embedding.unsqueeze(0).shape,gt_obj.unsqueeze(0).shape)
        # print(seg_embedding.unsqueeze(0).dtype,gt_obj.unsqueeze(0).dtype)
        if torch.isnan(seg_embedding).any() or torch.isinf(seg_embedding).any():
            raise ValueError("Found NaN or Inf in seg_embedding")
        ce_loss = cls_criterion(seg_embedding.unsqueeze(0),gt_obj.unsqueeze(0)).squeeze().mean()
        # ce_loss = ce_loss/torch.log(torch.tensor(num_classes))

        # ce_loss_seg= cls_criterion(segment_logit.unsqueeze(0),gt_obj.unsqueeze(0)).squeeze().mean()
        #Object Aware Prototype Contrastive Loss
        feat = clip_head(seg_embedding_dec)
        proto_cl_loss = cl_loss.contrastive_loss(feat,seg_embedding,viewpoint_cam.uid)

        # Photometric Loss  
        

        
        
        Ll1 =  l1_loss_mask(image, gt_image)
        loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))
        if iteration>2000:
            loss += 0.7*ce_loss
            loss += 0.7*proto_cl_loss
            # loss += 0.9*ce_loss_seg 
        
        """
        Within and Across class regularization
        """
        # Neighborhood for each gaussian
        # if iteration%500 ==0:
        #     near_index = gaussians.neighbor()
        #     _3d_feat,_3d_classes = gaussians.get_3d_classes_feature()
        #     opacity = gaussians.get_opacity
        #     L_smooth_feat = smooth_3d_intra_class(_3d_classes,_3d_feat,opacity,near_index)
        #     L_inter_class = smooth_3d_inter_class(_3d_classes,_3d_feat,opacity,near_index)
        #     loss += 0.03*L_smooth_feat 
        #     loss += 0.03*L_inter_class
        """
            Knowledge Distillation
        """   
        if iteration>4000 and iteration<7000:
            with torch.no_grad():
                feat = backbone_network_kd.feat(gt_image.unsqueeze(0)).squeeze(0)#.permute(2,0,1)
                h,w,e = feat.shape
                seg_embedding_dec = torch.nn.functional.interpolate(seg_embedding_dec.unsqueeze(0),(h,w),mode='bilinear', align_corners=False)
                seg_embedding_dec = seg_embedding_dec.squeeze(0).permute(1,2,0)
                
                #Huber Loss
                # kd_loss = torch.nn.functional.huber_loss(seg_embedding_dec, feat, delta=1.5, reduction="none")

                # Cosine Similarity Loss
                kd_loss = 1-torch.nn.functional.cosine_similarity(torch.nn.functional.normalize(seg_embedding_dec),torch.nn.functional.normalize(feat),dim=-1).mean()
                kd_loss = kd_loss.sum(dim=-1).nanmean()
                if  kd_loss.isnan():
                    kd_loss = torch.tensor(0.0, dtype=seg_embedding_dec.dtype, device=seg_embedding_dec.device)
                loss +=  0.1*kd_loss
            
                del kd_loss
        
        rendered_depth = render_pkg["depth"][0]
        midas_depth = torch.tensor(viewpoint_cam.depth_image).cuda()


        """
            TV Depth regularization based on segmentation 
            Note: Use target or rendered segmentation?
        """
        rendered_disparity = 1/(rendered_depth+1)

        loss += 0.06*second_order_tv(rendered_disparity,midas_depth,((iteration) / (opt.iterations)),num_classes,torch.argmax(seg_embedding,dim=0))
        loss += 0.07* second_order_tv(rendered_disparity,midas_depth,((iteration) / (opt.iterations)),num_classes)
        del seg_embedding, seg_embedding_dec

        rendered_depth = rendered_depth.reshape(-1, 1)
        midas_depth = midas_depth.reshape(-1, 1)

        depth_loss = min(
                        (1 - pearson_corrcoef( - midas_depth, rendered_depth)),
                        (1 - pearson_corrcoef(1 / (midas_depth + 200.), rendered_depth))
        )
        loss += args.depth_weight * depth_loss
        if iteration > args.end_sample_pseudo:
            args.depth_weight = 0.001       

        del midas_depth, feat, importance_map
        torch.cuda.empty_cache()

        # pseudo view training
        if iteration % args.sample_pseudo_interval == 0 and iteration > args.start_sample_pseudo and iteration < args.end_sample_pseudo:
            print("rendering pseudo images")
            if not pseudo_stack:
                pseudo_stack = scene.getPseudoCameras().copy()
            pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))
            render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background,lerfmodel=lerf_model)
            rendered_pseudo_image = render_pkg_pseudo["render"]
            rendered_depth_pseudo = render_pkg_pseudo["depth"][0]
            midas_depth_pseudo = estimate_depth(render_pkg_pseudo["render"], mode='train')

            rendered_depth_pseudo = rendered_depth_pseudo.reshape(-1, 1)
            midas_depth_pseudo = midas_depth_pseudo.reshape(-1, 1)
            depth_loss_pseudo = (1 - pearson_corrcoef(rendered_depth_pseudo, -midas_depth_pseudo)).mean()

            if torch.isnan(depth_loss_pseudo).sum() == 0:
                loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
                loss += loss_scale * args.depth_pseudo_weight * depth_loss_pseudo

            
            edge_image = np.array(rendered_pseudo_image.detach().cpu())
            edge_image =  np.moveaxis(edge_image, 0, -1)
            edge_image = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)
            edge_image = (255 * (edge_image - edge_image.min()) / (edge_image.max() - edge_image.min())).astype(np.uint8)
            edge_image = cv2.Canny(edge_image, low_threshold, high_threshold)
            edge_image = edge_image[:, :, None]
            edge_image = np.concatenate([edge_image, edge_image, edge_image], axis=2)
            edge_image = Image.fromarray(edge_image)

            control_image = estimate_depth(rendered_pseudo_image,mode='train')
            control_image = transform_to_pil(control_image).convert('RGB')
            with torch.no_grad(), torch.cuda.amp.autocast():
                result = gen_pipe(
                    
                    prompt = (
                    "ultra-realistic photo of the same indoor plant in a modern office space, "
                    "sharp focus, same lighting, clean background, consistent with original depth and layout"
                    ),
                    negative_prompt = (
                        "blurry, distorted, surreal, fantasy, overexposed, different object, cartoon"
                    ),
                    image=rendered_pseudo_image,
                    control_image=[control_image,edge_image],
                    num_inference_steps=30,
                    strength=0.25,   # allows image to be refined, not replaced
                    guidance_scale=6,
                )
            output_tensor = transform_to_tensor(result.images[0])
            
            if iteration%1000 ==0:
                file_name = str(iteration)+"_pseudo.png"
                transform_to_pil(rendered_pseudo_image).save(file_name)
                file_name = str(iteration)+"_generated.png"
                result.images[0].save(file_name)
              
            output_tensor = F.interpolate(
                output_tensor.unsqueeze(0), size=rendered_pseudo_image.shape[-2:], mode='bilinear', align_corners=False
            ).squeeze(0).cuda()
            Ll1 =  l1_loss_mask(rendered_pseudo_image, output_tensor)
            
            pseudo_loss = 0.4*((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(rendered_pseudo_image, output_tensor)))
            pseudo_loss.backward(retain_graph=True)
            del rendered_pseudo_image, control_image, edge_image, result, output_tensor
            torch.cuda.empty_cache()
        loss.backward()

        if iteration>2000:
            gradient_norm = torch.stack([torch.norm(p.grad) for p in lerf_model.parameters()])
            if not gradient_norm.mean().isnan():
                # torch.nn.utils.clip_grad_norm_(lerf_model.parameters(), 1.0)
                lerf_optimizer.step()
            else:
                print("~~~~~~~~~ Gradient_norm contains Nan, we will skip! gradient_norm: ",  gradient_norm)

            lerf_optimizer.zero_grad(set_to_none=True)
            lerf_opt_lr_scheduler.step()
            cls_decoder_optimizer.step()
            cls_decoder_optimizer.zero_grad(set_to_none=True)
            cls_head_optimizer.step()
            cls_head_optimizer.zero_grad(set_to_none=True)
            cls_clip_optimizer.step()
            cls_clip_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            testing_iterations, scene, lerf_model, render, (pipe, background))

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                torch.save(seg_decoder.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration),'seg_decoder.pth'))
                torch.save(seg_head.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration),'seg_head.pth'))
                torch.save(clip_head.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration),'clip_head.pth'))
                torch.save(lerf_model.state_dict(),os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration),'lerf_model.pth'))
            if iteration > first_iter and (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration > first_iter and (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Densification
            if  iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter,
                    render_pkg["pixels"],render_pkg['imgrad_pixels'], 
                    render_pkg["bg_pixels"],render_pkg["bg_imgrad_pixels"]
                )

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, iteration,args.boundary_threshold, args.size_threshold_rate)

                if iteration%opt.opacity_reset_interval == 0 or dataset.white_background == opt.densify_from_iter:
                    gaussians.reset_opacity()       

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
                # lerf_opt_lr_scheduler.step()
                # lerf_model.zero_grad()
                


            gaussians.update_learning_rate(iteration)
            if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
                    iteration > args.start_sample_pseudo:
                gaussians.reset_opacity()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene,lerf_model, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, lerfmodel=lerf_model, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 8):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()

                    _mask = None
                    _psnr = psnr(image, gt_image, _mask).mean().double()
                    _ssim = ssim(image, gt_image, _mask).mean().double()
                    _lpips = lpips(image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--opt_vlrenderfeat_from', type=int, default=True, help="Optimize the rendered VL feature map from the spec iteration. If it is < 0, then never optimize the feature map.")
    parser.add_argument('--fmap_resolution', type=int, default=-1, help="The scale for raysettings when render feature map. This can be -1, 4, 8 ...")
    parser.add_argument('--fmap_lr', type=float, default=1e-2, help="The learning rate for VL fmap")
    parser.add_argument('--fmap_render_radiithre', type=int, default=2, help="The radii threshold when rendering feature map.")
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--kd_backbone', type=str, default="dino_vitb16")
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_00, 20_00, 30_00, 50_00, 60_00, 80_00, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_00, 30_00, 50_00, 60_00, 70_00, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[50_00, 10_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--boundary_threshold", type=float, default=500)
    parser.add_argument("--size_threshold_rate", type=float, default=1)
    parser.add_argument("--train_bg", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print(args.test_iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")                       