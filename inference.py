# -*- coding: utf-8 -*-
from copyreg import pickle
import os
import sys
import os.path as osp
import cv2
import glob
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from scipy.spatial import cKDTree
from collections import defaultdict

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))
from lib.deca import DECA
from lib.datasets import datasets 
from lib.utils import util, detectors
from lib.utils.config import get_cfg_defaults, update_cfg


def reconstruct_and_render_dir(
    input_dir,
    output_dir,
    cfg_path,
    model_path,
    device="cuda",
    use_tex=False,
    vertex_colors=None,
):
    assert input_dir != output_dir, "Input and output directories are the same!"
    os.makedirs(output_dir, exist_ok=True)
    
    face_detector = detectors.iBugDetectors()
   
    # load test images
    test_dataset = datasets.TestData(
        input_dir, iscrop=True, face_detector=face_detector,
    )

    # load a configure file
    deca_cfg = get_cfg_defaults()
    update_cfg(deca_cfg, cfg_path)
    # update some config settings
    deca_cfg.model.use_tex = use_tex
    deca_cfg.rasterizer_type = "pytorch3d"
    deca_cfg.pretrained_modelpath = model_path
    # build a DECA model
    deca = DECA(config=deca_cfg)
    deca = deca.to(device)

    if vertex_colors is None:
        face_colors = None
    else:
        # 1 x NF x 3 x 3
        faces = deca.render.faces.to(device)
        colors = torch.tensor(vertex_colors, device=device).float() / 255.
        face_colors = util.face_vertices(colors, faces)
    
    pred_parameters = defaultdict(dict)
    for batch_data in tqdm(test_dataset):
        image_name = batch_data['imagename']
        image = batch_data['image'].to(device)[None,...]
        original_image = batch_data['original_image'][None, ...].to(device)
        tform = batch_data['tform'][None, ...]
        tform = torch.inverse(tform).transpose(1,2).to(device)
        batch_size = image.shape[0]

        with torch.no_grad():
            codedict = deca.encode(image)
            # store predict parameters
            for key in ['shape', 'tex', 'exp', 'pose', 'cam', 'light']:
                pred_parameters[image_name][key] = codedict[key].cpu().numpy()

            opdict = deca.decode(
                codedict,
                return_vis=False,
                vis_lmk=False,
            )

            points_scale = [deca.image_size, deca.image_size]
            _, _, h, w = original_image.shape
            trans_verts = util.transform_points(
                opdict["trans_verts"], tform, points_scale, [h, w],
            )

            ## render shape
            shape_images, _, grid, alpha_images = deca.render.render_shape(
                opdict["verts"],
                trans_verts,
                colors=face_colors,
                h=h,
                w=w,
                images=original_image,
                return_grid=True,
            )

            detail_normal_images = alpha_images * F.grid_sample(
                opdict["uv_detail_normals"], grid, align_corners=False,
            )

            shape_detail_images = deca.render.render_shape(
                opdict["verts"],
                trans_verts,
                colors=face_colors,
                detail_normal_images=detail_normal_images,
                h=h,
                w=w,
                images=original_image,
            )

            visdict = {
                "original_image": original_image,
                "shape_images": shape_images,
                "shape_detail_images": shape_detail_images,
            }
            
            image_grid = deca.visualize(visdict, size=(w+h)//2)

            cv2.imwrite(osp.join(output_dir, f"{image_name}.jpg"), image_grid)

    return pred_parameters


if __name__ == "__main__":
    asset_dir = osp.join(osp.dirname(__file__), "..", "models") 
    tdmm_type = "AR" # or "FLAME"
    
    # cfg_path = "/fsx/shiyangc/projects/DECA/configs/lightning/detail_final.yml"
    # model_path = "/fsx/shiyangc/checkpoints/deca_models/lightning/detail_final/last.ckpt"
    # cfg_path = "/fsx/shiyangc/projects/DECA/configs/lightning/mbv2_detail.yml"
    # model_path = "/fsx/shiyangc/checkpoints/deca_models/lightning/mbv2_detail/last.ckpt"
    cfg_path = "/fsx/shiyangc/checkpoints/deca_models/ar_multilinear/ar_mbv2_detail_newDyn_newLmkW_final/config.yaml"
    model_path = "/fsx/shiyangc/checkpoints/deca_models/ar_multilinear/ar_mbv2_detail_newDyn_newLmkW_final/last.ckpt"
    # cfg_path = "/fsx/shiyangc/checkpoints/deca_models/ar_multilinear/ar_detail_newDyn_newLmkW_rd1/config.yaml"
    # model_path = "/fsx/shiyangc/checkpoints/deca_models/ar_multilinear/ar_detail_newDyn_newLmkW_rd1/last.ckpt"

    highlight_parts = False
    if highlight_parts:
        dst_dir += "_parts_highlighted"
        # load all the templates
        if tdmm_type == "FLAME":
            verts_head, *_ = util.load_obj(osp.join(asset_dir, "flame/head_template.obj"))
            verts_eyebrows, *_ = util.load_obj(osp.join(asset_dir, "flame/head_template_eyebrows.obj"))
            verts_mouth, *_ = util.load_obj(osp.join(asset_dir, "flame/head_template_mouth.obj"))
        elif tdmm_type == "AR":
            verts_head, *_ = util.load_obj(osp.join(asset_dir, "ar_multilinear/base.obj"))
            verts_eyebrows, *_ = util.load_obj(osp.join(asset_dir, "ar_multilinear/eyebrows.obj"))
            verts_mouth, *_ = util.load_obj(osp.join(asset_dir, "ar_multilinear/lips.obj"))
        else:
            raise NotImplementedError(f"Unknown 3DMM type: {tdmm_type}!")

        # find indices for eyebrows and mouth
        tree = cKDTree(verts_head)
        _, indices_eyebrows = tree.query(verts_eyebrows)
        _, indices_mouth = tree.query(verts_mouth)
        # color different parts, 1 x NV x 3
        vertex_colors = 180. + np.zeros((1, verts_head.shape[0], 3))
        # black for eyebrows
        vertex_colors[:, indices_eyebrows] *= 0.3
        # slight dark gray for mouth
        vertex_colors[:, indices_mouth] *= 0.6
    else:
        vertex_colors = None
    
    for video_name in os.listdir(src_dir):
        input_dir = osp.join(src_dir, video_name)
        output_dir = osp.join(dst_dir, video_name)
        # reconstruct face using DECA
        pred_params = reconstruct_and_render_dir(
            input_dir,
            output_dir,
            cfg_path,
            model_path,
            vertex_colors=vertex_colors,
        )
