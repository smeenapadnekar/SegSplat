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
import copy
import matplotlib.pyplot as plt
import torch
from scene import Scene
import os
from tqdm import tqdm
import numpy as np
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2
import time
from tqdm import tqdm
import colorsys
from PIL import Image
from sklearn.decomposition import PCA

from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path, generate_spiral_path
from utils.general_utils import vis_depth
from lerf.lerf import LERFModel

def feature_to_rgb(features):
    # Input features shape: (16, H, W)
    
    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')

    return rgb_array

def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5

    
    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask

def render_set(model_path, name, iteration, views, gaussians, lerf_model,seg_decoder,seg_head, pipeline, background, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_feature16")
    gt_colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_objects_color")
    pred_obj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_pred")
    gts_path_raw = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_seg")
    pred_seg_path_raw = os.path.join(model_path, name, "ours_{}".format(iteration), "test_seg")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(gt_colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)
    makedirs(gts_path_raw, exist_ok=True)
    makedirs(pred_seg_path_raw, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background,lerfmodel=lerf_model)
        rendered_featmap = rendering["rendered_featmap"]
        seg_embedding = seg_decoder(rendered_featmap)
        # seg_embedding = clip_head(seg_embedding)
        seg_embedding = seg_head(seg_embedding)
        pred_obj = torch.argmax(seg_embedding,dim=0)
        pred_obj_mask_raw = (pred_obj.squeeze().cpu().numpy() * 255).astype(np.uint8)
        pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
        
        gt = view.original_image[0:3, :, :]
        gt_obj = view.objects.cuda().long().clamp(0,200)
        gt_obj_raw = (gt_obj.squeeze().cpu().numpy() * 255).astype(np.uint8)
        gt_rgb_mask = visualize_obj(gt_obj.cpu().numpy().astype(np.uint8))

        rgb_mask = feature_to_rgb(rendered_featmap)
        Image.fromarray(rgb_mask).save(os.path.join(colormask_path, view.image_name  + ".png"))
        Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, view.image_name  + ".png"))
        Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, view.image_name  + ".png"))

        Image.fromarray(gt_obj_raw).save(os.path.join(gts_path_raw, view.image_name  + ".png"))
        Image.fromarray(pred_obj_mask_raw).save(os.path.join(pred_seg_path_raw, view.image_name + ".png"))
        torchvision.utils.save_image(rendering["render"], os.path.join(render_path, view.image_name + '.png'))
                                            #'{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

        if args.render_depth:
            depth_map = vis_depth(rendering['depth'][0].detach().cpu().numpy())
            np.save(os.path.join(render_path, view.image_name + '_depth.npy'), rendering['depth'][0].detach().cpu().numpy())
            cv2.imwrite(os.path.join(render_path, view.image_name + '_depth.png'), depth_map)



def render_video(source_path, model_path, iteration, views, gaussians, pipeline, background, fps=30):
    render_path = os.path.join(model_path, 'video', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = copy.deepcopy(views[0])

    if source_path.find('llff') != -1:
        render_poses = generate_spiral_path(np.load(source_path + '/poses_bounds.npy'))
    elif source_path.find('360') != -1:
        render_poses = generate_ellipse_path(views)

    size = (view.original_image.shape[2], view.original_image.shape[1])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, fps, size)
    # final_video = cv2.VideoWriter(os.path.join('/ssd1/zehao/gs_release/video/', str(iteration), model_path.split('/')[-1] + '.mp4'), fourcc, fps, size)

    for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        rendering = render(view, gaussians, pipeline, background)

        img = torch.clamp(rendering["render"], min=0., max=1.)
        torchvision.utils.save_image(img, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        video_img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
        final_video.write(video_img)

    final_video.release()



def render_sets(dataset : ModelParams, pipeline : PipelineParams, args):

    with torch.no_grad():
        gaussians = GaussianModel(args)
        scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)

        num_classes = dataset.num_classes
        embedding_dim = 768
        print("num_class",num_classes)
        seg_decoder = torch.nn.Sequential(
                torch.nn.Conv2d(100, 1024, kernel_size=3, padding=1),  # 3x3 → padding=1
                torch.nn.ReLU(),
                torch.nn.Conv2d(1024, embedding_dim, kernel_size=3, padding=1),
                torch.nn.ReLU(),
            )

        clip_head = torch.nn.Conv2d(embedding_dim, 100, kernel_size=3, padding=1)

        seg_head = torch.nn.Sequential(
            torch.nn.Conv2d(embedding_dim, num_classes, kernel_size=1, padding=0),
            # torch.nn.Softmax()  # make sure softmax is over class dimension
        )
        lerf_model = LERFModel()
        lerf_model.cuda()
        
        seg_decoder.cuda()
        clip_head.cuda()
        seg_head.cuda()
        seg_decoder.eval()
        clip_head.eval()
        seg_head.eval()
        
        seg_decoder.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"seg_decoder.pth")))
        seg_head.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"seg_head.pth")))
        clip_head.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"clip_head.pth")))
        lerf_model.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"lerf_model.pth")))      
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if args.video:
            render_video(dataset.source_path, dataset.model_path, scene.loaded_iter, scene.getTestCameras(),
                         gaussians, pipeline, background, args.fps)

        if not args.skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, lerf_model, seg_decoder, seg_head, pipeline, background, args)
        if not args.skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, lerf_model, seg_decoder, seg_head, pipeline, background, args)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--render_depth", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), pipeline.extract(args), args)
