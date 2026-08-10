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
import matplotlib.pyplot as plt
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from simple_diff_gaussian_rasterization import GaussianRasterizer as SimpleGaussianRasterizer

from enum import Enum

class LERFFieldHeadNames(Enum):
    """Possible field outputs"""
    HASHGRID = "hashgrid"
    CLIP = "clip"
    DINO = "dino"
    CLIP_HYBRID = "clip_hybrid"


def render(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0,
           override_color = None, lerfmodel=None, bvl_feature_precomp=False, fmap_resolution=-1, 
           fmap_render_radiithre=2,importance_map=None,bg_map=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # default
    if importance_map is None:
        importance_map = torch.zeros(int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)).cuda()
    if bg_map is None:
        bg_map = torch.zeros(int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)).cuda()

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if min(pc.bg_color.shape) != 0:
        bg_color = torch.tensor([0., 0., 0.]).cuda()

    confidence = pc.confidence if pipe.use_confidence else torch.ones_like(pc.confidence) 
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = bg_color, #torch.tensor([1., 1., 1.]).cuda() if white_bg else torch.tensor([0., 0., 0.]).cuda(), #bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        confidence=confidence
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color


    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth, alpha,  pixels, bg_pixels, imgrad_pixels, bg_imgrad_pixels = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        importance_map = importance_map,
        bg_map = bg_map)

    # print(pixels[:10],imgrad_pixels[:100])
    # rendered_image_list, depth_list, alpha_list = [], [], []
    # for i in range(5):
    #     rendered_image, radii, depth, alpha = rasterizer(
    #         means3D=means3D,
    #         means2D=means2D,
    #         shs=shs,
    #         colors_precomp=colors_precomp,
    #         opacities=opacity,
    #         scales=scales,
    #         rotations=rotations,
    #         cov3D_precomp=cov3D_precomp)
    #     rendered_image_list.append(rendered_image)
    #     depth_list.append(depth)
    #     alpha_list.append(alpha)
    # def mean1(t):
    #     return torch.mean(torch.stack(t), 0)
    # rendered_image, depth, alpha = mean1(rendered_image_list), mean1(depth_list), mean1(alpha_list)
    
    sem_class = pc.get_sem_class
    rendered_featmap = None
    rendered_featmap_ex = None
    # print(opacity.shape,sem_class.shape)
    valid_gaussian_mask = (opacity >= 0.1).squeeze(-1).detach() & (radii > fmap_render_radiithre).detach()
    # valid_gaussian_mask = (opacity >= 0).squeeze(-1).detach() & (radii > fmap_render_radiithre).detach()
    clip_scales = radii[valid_gaussian_mask].detach() # TODO: the value of clip_scales needs to be checked.
    lerf_field_outputs = lerfmodel(pc, clip_scales, valid_gaussian_mask) # dict with keys "LERFFieldHeadNames.HASHGRID, LERFFieldHeadNames.CLIP, LERFFieldHeadNames.DINO"

    feature_clipmap_precomp = lerf_field_outputs[LERFFieldHeadNames.CLIP.value]
    # print("size of feat",feature_clipmap_precomp.shape, valid_gaussian_mask.any())
    simple_rasterizer = SimpleGaussianRasterizer(raster_settings=raster_settings)
    cov3D_precomp_ = cov3D_precomp_[valid_gaussian_mask].detach() if cov3D_precomp is not None else None
    rendered_featmap, rendered_featmap_ex, radii_rendered_featmap = simple_rasterizer(
        means3D=means3D[valid_gaussian_mask].detach(),
        means2D=means2D[valid_gaussian_mask].detach(),
        shs=None,
        colors_precomp=feature_clipmap_precomp.float(),
        colors_ex_precomp=sem_class[valid_gaussian_mask].float(),
        opacities=opacity[valid_gaussian_mask].detach(),
        scales=scales[valid_gaussian_mask].detach(),
        rotations=rotations[valid_gaussian_mask].detach(),
        cov3D_precomp=cov3D_precomp_)
    # print("rendered_feature map",rendered_featmap_ex.shape)
    # if min(pc.bg_color.shape) != 0:
    #     print("im in hell")
    #     rendered_image = rendered_image + (1 - alpha) * torch.sigmoid(pc.bg_color)  # torch.ones((3, 1, 1)).cuda()
    
    # print(rendered_featmap.shape,rendered_featmap_ex.shape)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth,
            "pixels": pixels,
            "imgrad_pixels": imgrad_pixels,
            "bg_pixels": bg_pixels,
            "bg_imgrad_pixels": bg_imgrad_pixels,
            "rendered_featmap": rendered_featmap,
            "segment_logit": rendered_featmap_ex}
