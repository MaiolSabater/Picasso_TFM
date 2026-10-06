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

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, StyleOptimizationParams
from utils.nnfm_loss import NNFMLoss, match_colors_for_image_set, color_histgram_match
from utils.image_utils import load_and_preprocess_style_image
from utils.group_theme_recolor_new import recolor_scene_to_style, GT_MODES
import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def apply_param_freezing(gaussians, freeze_sh0, freeze_sh_rest, freeze_opacity,
                         freeze_geometry, freeze_xyz=False, freeze_scale=False, freeze_rotation=False):
    gaussians._features_dc.requires_grad_(not freeze_sh0)
    gaussians._features_rest.requires_grad_(not freeze_sh_rest)
    gaussians._opacity.requires_grad_(not freeze_opacity)
    gaussians._xyz.requires_grad_(not (freeze_geometry or freeze_xyz))
    gaussians._scaling.requires_grad_(not (freeze_geometry or freeze_scale))
    gaussians._rotation.requires_grad_(not (freeze_geometry or freeze_rotation))


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    nnfm_loss_fn = NNFMLoss(device='cuda')
    if args.point_cloud:
        xyz, o, s = gaussians.load_ply(args.point_cloud, reset_basis_dim=args.reset_basis_dim)
        original_xyz, original_opacity, original_scale = torch.tensor(xyz).cuda(), torch.tensor(o).cuda(), torch.tensor(s).cuda()
        first_iter = 30_000

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # Apply parameter freezing before the first render so no gradient leaks through on iter 1
    apply_param_freezing(gaussians, args.freeze_sh0, args.freeze_sh_rest, args.freeze_opacity,
                         args.freeze_geometry, args.freeze_xyz, args.freeze_scale, args.freeze_rotation)

    # Load style images
    style_img = load_and_preprocess_style_image(args.style, (scene.img_width, scene.img_height))
    # choose other hues:
    if args.second_style:
        style_img2 = imageio.imread(args.second_style, pilmode="RGB").astype(np.float32) / 255.0
        style_img2 = cv2.resize(style_img2, (style_img.shape[1],style_img.shape[0]), interpolation=cv2.INTER_AREA)
        style_img2 = torch.from_numpy(style_img2).cuda()

    # prepare depth image
    depth_img_list = []
    # mask for spatial control
    mask_img_list = []
    mask_half_list = []
    with torch.no_grad():
        for i, view in enumerate(tqdm(scene.getTrainCameras(), desc="Rendering original depth")):
            depth_render = render(view, gaussians, pipe, background)["depth"]
            depth_img_list.append(depth_render)

            if args.mask_dir:
                select_mask = np.load(os.path.join(args.mask_dir, f'{view.image_name[:-4]}.npy'))
                select_mask = gaussian_filter(select_mask, sigma=1)
                
                mask_img_list.append(torch.from_numpy(cv2.resize(select_mask.astype(np.uint8), (scene.img_width, scene.img_height),interpolation=cv2.INTER_AREA)))
                mask_half_list.append(torch.from_numpy(cv2.resize(select_mask.astype(np.uint8), (scene.img_width//2, scene.img_height//2),interpolation=cv2.INTER_AREA)))
    # precolor step
    if not args.preserve_color:
        gt_img_list = []
        for view in scene.getTrainCameras():
            gt_img_list.append(view.original_image.permute(1,2,0))
        gt_imgs = torch.stack(gt_img_list)

        if not args.mask_dir:
            print('Changing Colors...')
            target_style_img = style_img if not args.second_style else style_img2
            V = gt_imgs.shape[0]
            include = None
            if 0 < args.gt_theme_views < V:
                # The group theme (Sec 3.3) is a global m-colour summary, so it
                # doesn't need every training view to converge -- subsample the
                # views that DEFINE it (all views are still recoloured against
                # the result) to keep optimize_group_theme's EM loop from
                # scaling with the full training-view count.
                idx = np.round(np.linspace(0, V - 1, num=args.gt_theme_views)).astype(int)
                include = sorted(set(idx.tolist()))
            
            import json
            manual_colors = None
            if args.manual_colors:
                manual_colors = {int(k): v for k, v in json.loads(args.manual_colors).items()}
            manual_map = None
            if args.manual_map:
                manual_map = {int(k): int(v) for k, v in json.loads(args.manual_map).items()}

            if args.recolor_method == "rgbxy":
                from utils.rgbxy_recolor import rgbxy_recolor_gaussians, rgbxy_recolor_image
                camera_centers = None
                if args.recolor_space == "rgb_azimuth":
                    camera_centers = np.stack([
                        c.camera_center.detach().cpu().numpy()
                        for c in scene.getTrainCameras()
                    ])
                new_dc_rgb, scene_palette, new_palette = rgbxy_recolor_gaussians(
                    gaussians, target_style_img, gt_imgs=gt_imgs,
                    n_colors=args.n_colors, style_n_colors=args.style_n_colors,
                    space=args.recolor_space, xy_weight=args.xy_weight,
                    up_axis=args.up_axis,
                    azimuth_use_zenith=args.azimuth_zenith,
                    azimuth_use_radius=args.azimuth_radius,
                    camera_centers=camera_centers,
                    debug=args.recolor_debug, output_path=args.model_path,
                    palette_source="images",
                    return_palettes=True,          # Option A: need the palettes for images
                )
                # Option A: recolour the training images with the SAME palette mapping
                imgs_np = gt_imgs.detach().cpu().numpy().astype(np.float64)   # (V,H,W,3)
                new_imgs = np.empty_like(imgs_np)
                for v in range(len(imgs_np)):
                    new_imgs[v] = rgbxy_recolor_image(imgs_np[v], scene_palette, new_palette)
                gt_imgs = torch.from_numpy(new_imgs).to(dtype=gt_imgs.dtype, device=gt_imgs.device)
            else:
                print('ZZZZZZZZZZZZZ')
                gt_imgs, new_dc_rgb = recolor_scene_to_style(
                    gaussians, gt_imgs, target_style_img,
                    m=args.gt_theme_size, mode="min_reduction",
                    hue_tol=args.gt_hue_tol, power=args.gt_power,
                    max_pixels=args.gt_max_pixels, seed=args.gt_seed, debug=args.recolor_debug,
                    include=include, n_jobs=args.gt_n_jobs, l_strength=args.gt_l_strength, output_path=args.model_path,
                    theme_source=args.gt_theme_source, recolor_space=args.gt_recolor_space,
                    margin_pull=args.margin_pull, margin_base_scale=args.margin_base_scale, manual_theme_colors=manual_colors, direct_match=args.direct_match,
                    match_by=args.gt_match_by, unique_by=args.gt_unique_by
                )
            gaussians.apply_palette_recolor(new_dc_rgb)
            print('Recoloring applied.')
        else:
            mask_imgs = torch.stack(mask_img_list).unsqueeze(-1).repeat(1,1,1,3).cuda()
            if args.second_style:
                gt_imgs1, color_ct = color_histgram_match(gt_imgs, style_img)
                gt_imgs2, color_ct2 = color_histgram_match(gt_imgs, style_img2)
                gt_imgs = gt_imgs1 * (1-mask_imgs) + gt_imgs2 * mask_imgs
            else:
                recolor_gt_imgs, color_ct = color_histgram_match(gt_imgs, style_img)
                gt_imgs = recolor_gt_imgs * mask_imgs + gt_imgs * (1-mask_imgs)


        gt_img_list = [item.permute(2,0,1) for item in gt_imgs]
        imageio.imwrite(
            os.path.join(args.model_path, "gt_image_recolor.png"),
            np.clip(gt_img_list[0].permute(1,2,0).detach().cpu().numpy() * 255.0, 0.0, 255.0).astype(np.uint8),
        )

    with torch.no_grad():
        preview_view = scene.getTrainCameras()[0]
        preview_render = render(preview_view, gaussians, pipe, background)["render"]
        imageio.imwrite(
            os.path.join(args.model_path, "preview_render.png"),
            np.clip(preview_render.permute(1,2,0).detach().cpu().numpy() * 255.0, 0.0, 255.0).astype(np.uint8),
        )

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            depth_stack = depth_img_list.copy()
            if not args.preserve_color:
                gt_stack = gt_img_list.copy()
            if args.mask_dir:
                mask_stack = mask_half_list.copy()
        view_idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(view_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, depth_image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg['depth'], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        if not args.preserve_color:
            gt_image = gt_stack.pop(view_idx).cuda()
        else:
            gt_image = viewpoint_cam.original_image.cuda()
        depth_gt = depth_stack.pop(view_idx)
        mask_image = None

        gt_image = gt_image.unsqueeze(0)
        image = image.unsqueeze(0)

        if args.mask_dir:
            loss_type = ['spatial_loss','content_loss'] if not args.second_style else ['spatial_loss','nnfm_loss','content_loss']
            mask_image = mask_stack.pop(view_idx).cuda()
        elif args.preserve_color or (args.second_style and not args.mask_dir):
            loss_type = ['lum_nnfm_loss','content_loss']
        elif args.scale_level is not None:
            loss_type = ["scale_loss", "content_loss"]
        else:
            loss_type = ['lum_nnfm_loss','content_loss']


        if iteration > first_iter + 200: # stylization
            apply_param_freezing(gaussians, args.freeze_sh0, args.freeze_sh_rest, args.freeze_opacity,
                                 args.freeze_geometry, args.freeze_xyz, args.freeze_scale, args.freeze_rotation)
            loss_dict = nnfm_loss_fn(
                F.interpolate(
                    image,
                    size=None,
                    scale_factor=0.5,
                    mode="bilinear",
                ),
                style_img.permute(2,0,1).unsqueeze(0),
                blocks=args.vgg_block,
                loss_names=loss_type,
                contents=F.interpolate(
                    gt_image,
                    size=None,
                    scale_factor=0.5,
                    mode="bilinear",
                ),
                scale_level=args.scale_level,
                x_mask=mask_image,
                styles2=style_img2.permute(2,0,1).unsqueeze(0) if args.second_style else None,
            )
            image.requires_grad_(True)
            w_variance = torch.mean(torch.pow(image[:, :, :, :-1] - image[:, :, :, 1:], 2))
            h_variance = torch.mean(torch.pow(image[:, :, :-1, :] - image[:, :, 1:, :], 2))

            loss_dict[loss_type[0]] *= args.style_weight
            loss_dict["content_loss"] *= args.content_weight
            loss_dict["img_tv_loss"] = args.img_tv_weight * (h_variance + w_variance) / 2.0
            loss_dict['depth_loss'] = l2_loss(depth_image, depth_gt)
            
        else:
            apply_param_freezing(gaussians, args.freeze_sh0, args.freeze_sh_rest, args.freeze_opacity,
                                 args.freeze_geometry, args.freeze_xyz, args.freeze_scale, args.freeze_rotation)
            loss_dict = {}
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            loss_dict['ddsm_loss'] = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # opacity & scale regulariers
        loss_dict['opacity_regu'] = l1_loss(gaussians._opacity, original_opacity)
        loss_dict['scale_regu'] = l1_loss(gaussians._scaling, original_scale)

        loss = sum(list(loss_dict.values()))

        loss.backward()

        iter_end.record()
        if iteration % 200 == 0:
            with torch.no_grad():
                prev = render(scene.getTrainCameras()[0], gaussians, pipe, background)["render"]
                imageio.imwrite(
                    os.path.join(args.model_path, f"preview_{iteration:06d}.png"),
                    np.clip(prev.permute(1,2,0).detach().cpu().numpy()*255, 0, 255).astype(np.uint8))
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = StyleOptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    # style params
    parser.add_argument("--point_cloud", type=str, help='trained real 3DGS ply', default = None)
    parser.add_argument("--style", type=str, help="path to style image")
    parser.add_argument("--second_style", type=str, default="", help="path to second style image")
    parser.add_argument("--style_weight", type=float, default=5, help="style loss weight")
    parser.add_argument("--content_weight", type=float, default=5e-3, help="content loss weight")
    parser.add_argument("--img_tv_weight", type=float, default=1, help="image tv loss weight")
    parser.add_argument(
        "--vgg_block",
        type=list,
        default=[2,3],
        help="vgg block for nnfm extracting feature maps",
    )
    parser.add_argument(
        "--reset_basis_dim",
        type=int,
        default=1,
        help="whether to reset the number of spherical harmonics basis to this specified number",
    )
    parser.add_argument("--preserve_color", action="store_true", default=False)
    parser.add_argument("--scale_level", type=int, default=None, choices=[0,1,2], help='the scale of style pattern, can be [0,1,2]')
    parser.add_argument("--mask_dir", default=None, type=str, help="The directory of multiview masks")

    # Group-Theme Recoloring precoloring params (used when mask_dir is not set)
    parser.add_argument("--gt_theme_size", type=int, default=5, help="Group-Theme recoloring: number of theme colors (paper's m)")
    parser.add_argument("--gt_mode", type=str, default="kmeans", choices=list(GT_MODES.keys()), help="Group-Theme recoloring: assignment mode (Sec 3.3)")
    parser.add_argument("--gt_hue_tol", type=float, default=18.0, help="Group-Theme recoloring: hue tolerance in degrees for matching the style palette (Sec 3.4). Only has an effect when --gt_match_by hue.")
    parser.add_argument("--gt_match_by", choices=["unique", "hue", "ab"], default="unique",
        help="Group-Theme recoloring: how each scene theme colour is matched to a style palette colour. "
             "'unique' (default): one-to-one Hungarian assignment. 'hue': per-colour nearest-hue-within-tolerance "
             "match (not one-to-one, multiple scene colours can collapse onto the same style colour). "
             "'ab': per-colour nearest style colour in ab (not one-to-one either).")
    parser.add_argument("--gt_unique_by", choices=["ab", "hue"], default="hue",
        help="Group-Theme recoloring: cost metric for --gt_match_by unique's Hungarian assignment.")
    parser.add_argument("--gt_power", type=float, default=2.0, help="Group-Theme recoloring: IDW power for propagating palette shifts (Sec 3.5)")
    parser.add_argument("--gt_max_pixels", type=int, default=50_000, help="Group-Theme recoloring: per-image pixel subsample cap for palette extraction")
    parser.add_argument("--gt_theme_views", type=int, default=30, help="Group-Theme recoloring: max number of training views used to DEFINE the group theme (Sec 3.3); every view is still recoloured against it. Set <=0 to use every training view.")
    parser.add_argument("--gt_n_jobs", type=int, default=-1, help="Group-Theme recoloring: worker processes for per-image palette extraction (Sec 3.2). -1 = all cores, 1 = sequential.")
    parser.add_argument("--gt_seed", type=int, default=0, help="Group-Theme recoloring: RNG seed for subsampling/clustering")
    parser.add_argument("--gt_l_strength", type=float, default=0.0)
    parser.add_argument("--gt_theme_source", type=str, default="images")
    parser.add_argument("--recolor_method", choices=["grouptheme", "rgbxy"], default="grouptheme")
    parser.add_argument("--gt_recolor_space", choices=["ab", "lab"], default="ab")
    parser.add_argument("--manual_colors", type=str, default="",
        help='JSON idx->rgb, e.g. \'{"0": [0.1,0.2,0.9], "2": [230,130,0]}\'. '
            'For grouptheme: theme index. For rgbxy: scene palette index.')
    parser.add_argument("--manual_map", type=str, default="",
        help='JSON idx->style_idx, e.g. \'{"0": 3}\'. Maps a source palette entry '
            'to an existing style palette colour by index.')

    parser.add_argument("--direct_match", action="store_true",
        help="skip group-theme; match scene palette directly to style palette")
    parser.add_argument("--recolor_debug", action="store_true", default=False,
        help="save recolor debug plots (palette_mapping.png, ab_recolor.png, etc.)")

    # parameter freeze flags for ablation experiments
    parser.add_argument("--freeze_sh0", action="store_true", default=False, help="Freeze 0-degree SH coefficients")
    parser.add_argument("--freeze_sh_rest", action="store_true", default=False, help="Freeze higher-degree SH coefficients")
    parser.add_argument("--freeze_opacity", action="store_true", default=False, help="Freeze opacity")
    parser.add_argument("--freeze_geometry", action="store_true", default=False, help="Freeze xyz, scaling, and rotation (shorthand)")
    parser.add_argument("--freeze_xyz", action="store_true", default=False, help="Freeze centroid positions")
    parser.add_argument("--freeze_scale", action="store_true", default=False, help="Freeze Gaussian scales")
    parser.add_argument("--freeze_rotation", action="store_true", default=False, help="Freeze Gaussian rotations (quaternions)")
    parser.add_argument("--precolor_only", action="store_true", default=False, help="Always write the precoloring color transform to SH0/SH_rest, even if they are frozen for training")

    parser.add_argument("--margin_pull", action="store_true",
                    help="adaptive margin-pull toward assigned theme target after IDW")
    parser.add_argument("--margin_base_scale", type=float, default=0.5,
                        help="margin-pull radius scale (smaller = tighter clamp)")


    parser.add_argument("--recolor_space", default="rgb",
                    choices=["rgb", "rgbxy", "rgb_azimuth"],
                    help="feature space for the rgbxy method")
    parser.add_argument("--xy_weight", type=float, default=1.0,
                        help="scale on the spatial/angular block (rgbxy/azimuth)")
    parser.add_argument("--up_axis", type=int, default=1,
                        help="vertical axis for azimuth (2=Z, 1=Y). Match your scene.")
    parser.add_argument("--azimuth_zenith", action="store_true",
                        help="add elevation dim to rgb_azimuth")
    parser.add_argument("--azimuth_radius", action="store_true",
                        help="add distance dim to rgb_azimuth")
    parser.add_argument("--n_colors", type=int, default=6,
                        help="scene palette size (>=6 for 5D azimuth/rgbxy)")
    parser.add_argument("--style_n_colors", type=int, default=6)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
