import os
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from yacs.config import CfgNode
from .models.encoders import ResnetEncoder, MobilenetV2Encoder
from .models.FLAME import FLAME, FLAMETex
from .models.ar_multilinear import ARMultilinear, ARMultilinearTex
from .models.decoders import Generator
from .utils import util
from .utils.renderer import (
    SRenderY, 
    set_rasterizer,
)

torch.backends.cudnn.benchmark = True


class DECA(nn.Module):
    def __init__(self, config: CfgNode):
        super(DECA, self).__init__()
        self.config = config
        self.use_tex = self.config.model.use_tex
        self.uv_size = self.config.model.uv_size
        self.image_size = self.config.dataset.image_size

        self._create_model(self.config.model)
        self._setup_renderer(self.config.model)

    def _setup_renderer(self, model_cfg):
        # face mask for rendering details
        mask = cv2.imread(model_cfg.face_eye_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.register_buffer(
            'uv_face_eye_mask', 
            F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]),
        )
        # displacement correction
        fixed_dis = np.load(model_cfg.fixed_displacement_path)
        self.register_buffer('fixed_uv_dis', torch.tensor(fixed_dis).float())
        # dense mesh template, for save detail mesh
        self.dense_template = np.load(
            model_cfg.dense_template_path, allow_pickle=True, encoding='latin1',
        ).item()

        set_rasterizer(self.config.rasterizer_type)
        self.render = SRenderY(
            self.image_size,
            model_cfg,
            rasterizer_type=self.config.rasterizer_type,
        )

    def _create_model(self, model_cfg):
        # set up parameters
        self.n_detail = model_cfg.n_detail
        if model_cfg.tdmm_type == "AR":
            # expression params only
            self.n_cond = model_cfg.n_exp
        else:
            # exp + jaw pose
            self.n_cond = model_cfg.n_exp + 3
        
        self.num_list = [
            model_cfg.n_shape,
            model_cfg.n_tex,
            model_cfg.n_exp,
            model_cfg.n_pose,
            model_cfg.n_cam,
            model_cfg.n_light,
        ]
        self.n_param = sum(self.num_list)
        self.param_dict = {key: model_cfg.get('n_' + key) for key in model_cfg.param_list}
        
        # encoders with different options for backbone
        if model_cfg.backbone == "resnet50":
            self.E_flame = ResnetEncoder(outsize=self.n_param)
            self.E_detail = ResnetEncoder(outsize=self.n_detail) 
        elif model_cfg.backbone == "mobilenetv2":
            self.E_flame = MobilenetV2Encoder(outsize=self.n_param)
            self.E_detail = MobilenetV2Encoder(outsize=self.n_detail)
        else:
            raise NotImplementedError(f"Unknown backbone {model_cfg.backbone}")
        
        # decoders
        if model_cfg.tdmm_type == "AR":
            self.flame = ARMultilinear(model_cfg)
            if self.use_tex:
                self.flametex = ARMultilinearTex(model_cfg)
        else:
            self.flame = FLAME(model_cfg)
            if self.use_tex:
                self.flametex = FLAMETex(model_cfg)
        
        self.D_detail = Generator(
            latent_dim=self.n_detail + self.n_cond,
            out_channels=1,
            out_scale=model_cfg.max_z,
            sample_mode='bilinear',
        )

        # load pre-trained model
        model_path = self.config.pretrained_modelpath
        if os.path.exists(model_path):
            print(f'Find and load pre-trained model: {model_path}')
            checkpoint = torch.load(model_path)
            util.copy_state_dict(
                self.E_flame.state_dict(),
                checkpoint["state_dict"],
                prefix="deca.E_flame.",
            )
            util.copy_state_dict(
                self.E_detail.state_dict(),
                checkpoint["state_dict"],
                prefix="deca.E_detail.",
            )
            util.copy_state_dict(
                self.D_detail.state_dict(),
                checkpoint["state_dict"],
                prefix="deca.D_detail.",
            )
        else:
            print(f'Model not found, please check {model_path}')
        # eval mode
        self.E_flame.eval()
        self.E_detail.eval()
        self.D_detail.eval()

    def decompose_code(self, code, num_dict):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start + int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict

    def visofp(self, normals):
        ''' visibility of keypoints, based on the normal direction
        '''
        normals68 = self.flame.seletec_3d68(normals)
        vis68 = (normals68[:,:,2:] < 0.1).float()
        return vis68

    def encode(self, images, use_detail=True):
        if use_detail:
            # use_detail is for training detail model, need to set coarse model as eval mode
            with torch.no_grad():
                parameters = self.E_flame(images)
        else:
            parameters = self.E_flame(images)
        codedict = self.decompose_code(parameters, self.param_dict)
        codedict['images'] = images

        # For AR multilinear model, we need to clamp the expression code to [0, 1]
        if self.config.model.tdmm_type == "AR":
            codedict['exp'] = torch.clamp(codedict['exp'], min=0.0, max=1.0)

        if use_detail:
            detailcode = self.E_detail(images)
            codedict['detail'] = detailcode

        return codedict

    def decode(
        self,
        codedict,
        use_detail=True,
        iddict=None,
        rendering=True,
        vis_lmk=True,
        return_vis=True,
        render_orig=False,
        original_image=None,
        tform=None,
    ):
        images = codedict['images']
        batch_size = images.shape[0]
        
        ## decode
        verts, landmarks2d, landmarks3d = self.flame(
            shape_params=codedict['shape'],
            expression_params=codedict['exp'],
            pose_params=codedict['pose'],
        )

        if self.use_tex:
            albedo = self.flametex(codedict['tex'])
        else:
            albedo = torch.zeros(
                [batch_size, 3, self.uv_size, self.uv_size],
                device=images.device,
            )
            
        landmarks3d_world = landmarks3d.clone()

        ## projection
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[..., :2]
        landmarks2d[..., 1:] = -landmarks2d[..., 1:]
        
        landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam'])
        landmarks3d[..., 1:] = -landmarks3d[..., 1:]
        
        trans_verts = util.batch_orth_proj(verts, codedict['cam'])
        trans_verts[..., 1:] = -trans_verts[..., 1:]
        opdict = {
            'verts': verts,
            'trans_verts': trans_verts,
            'landmarks2d': landmarks2d,
            'landmarks3d': landmarks3d,
            'landmarks3d_world': landmarks3d_world,
            'albedo': albedo,
        }

        ## rendering
        if return_vis and render_orig and original_image is not None and tform is not None:
            points_scale = [self.image_size, self.image_size]
            _, _, h, w = original_image.shape
            trans_verts = util.transform_points(trans_verts, tform, points_scale, [h, w])
            landmarks2d = util.transform_points(landmarks2d, tform, points_scale, [h, w])
            landmarks3d = util.transform_points(landmarks3d, tform, points_scale, [h, w])
            background = original_image
            images = original_image
        else:
            h, w = self.image_size, self.image_size
            background = None

        if rendering:
            # this part is different from the official version
            ops = self.render(verts, trans_verts, albedo, lights=codedict['light'])
            ## output
            opdict['grid'] = ops['grid']
            opdict['rendered_images'] = ops['images']
            opdict['alpha_images'] = ops['alpha_images']
            opdict['normal_images'] = ops['normal_images']
        
        if use_detail:
            if iddict is None:
                if self.config.model.tdmm_type == "AR":
                    # pose parameters in AR MultiLinear model do not contain jaw pose
                    d_code = torch.cat([codedict['exp'], codedict['detail']], dim=1)
                else:
                    d_code = torch.cat([codedict['pose'][:, 3:], codedict['exp'], codedict['detail']], dim=1)
            else:
                if self.config.model.tdmm_type == "AR":
                    # pose parameters in AR MultiLinear model do not contain jaw pose
                    d_code = torch.cat([iddict['exp'], codedict['detail']], dim=1)
                else:
                    d_code = torch.cat([iddict['pose'][:, 3:], iddict['exp'], codedict['detail']], dim=1)
            uv_z = self.D_detail(d_code)
            uv_detail_normals = self.render.displacement2normal(uv_z, verts, ops['normals'])
            uv_shading = self.render.add_SHlight(uv_detail_normals, codedict['light'])
            uv_texture = albedo * uv_shading

            opdict['uv_texture'] = uv_texture 
            opdict['normals'] = ops['normals']
            opdict['uv_detail_normals'] = uv_detail_normals
            opdict['displacement_map'] = uv_z + self.fixed_uv_dis[None, None, ...]
        else:
            if rendering:
                uv_coarse_normals = self.render.world2uv(ops['normals'])
                uv_shading = self.render.add_SHlight(uv_coarse_normals, codedict['light'])
                uv_texture = albedo * uv_shading
        
        if vis_lmk:
            landmarks3d_vis = self.visofp(ops['transformed_normals'])
            landmarks3d = torch.cat([landmarks3d, landmarks3d_vis], dim=2)
            opdict['landmarks3d'] = landmarks3d

        if return_vis:
            ## render shape
            shape_images, _, grid, alpha_images = self.render.render_shape(
                verts, trans_verts, h=h, w=w, images=background, return_grid=True,
            )

            visdict = {
                'inputs': images, 
                'landmarks2d': util.tensor_vis_landmarks(images, landmarks2d),
                'landmarks3d': util.tensor_vis_landmarks(images, landmarks3d),
                'shape_images': shape_images,
            }
            ## extract texture
            ## TODO: current resolution 256x256, support higher resolution, and add visibility
            uv_pverts = self.render.world2uv(trans_verts)
            
            uv_gt = F.grid_sample(
                images, uv_pverts.permute(0, 2, 3, 1)[..., :2], mode='bilinear', align_corners=False,
            )
            
            if self.use_tex:
                visdict['rendered_images'] = ops['images']
                ## TODO: poisson blending should give better-looking results
                uv_texture_gt = \
                    uv_gt[:, :3] * self.uv_face_eye_mask + \
                    uv_texture[:, :3] * (1. - self.uv_face_eye_mask)
            else:
                uv_texture_gt = \
                    uv_gt[:, :3] * self.uv_face_eye_mask + \
                    torch.ones_like(uv_gt[:, :3]) * (1. - self.uv_face_eye_mask) * 0.7
            
            opdict['uv_texture_gt'] = uv_texture_gt

            if use_detail:
                detail_normal_images = alpha_images * F.grid_sample(
                    uv_detail_normals, grid, align_corners=False,
                )
                shape_detail_images = self.render.render_shape(
                    verts,
                    trans_verts,
                    detail_normal_images=detail_normal_images,
                    h=h,
                    w=w,
                    images=background,
                )
                visdict['shape_detail_images'] = shape_detail_images

            return opdict, visdict
        else:
            return opdict

    def decode_lite(self, codedict, use_detail=True):
        images = codedict['images']
        batch_size = images.shape[0]

        ## decode
        verts, landmarks2d, landmarks3d = self.flame(
            shape_params=codedict['shape'],
            expression_params=codedict['exp'],
            pose_params=codedict['pose'],
        )
        landmarks3d_world = landmarks3d.clone()

        ## projection
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[..., :2]
        landmarks2d[..., 1:] = -landmarks2d[..., 1:]
        landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam'])
        landmarks3d[..., 1:] = -landmarks3d[..., 1:]
        trans_verts = util.batch_orth_proj(verts, codedict['cam'])
        trans_verts[..., 1:] = -trans_verts[..., 1:]

        opdict = {
            'verts': verts,
            'trans_verts': trans_verts,
            'landmarks2d': landmarks2d,
            'landmarks3d': landmarks3d,
            'landmarks3d_world': landmarks3d_world,
        }

        if use_detail:
            if self.config.model.tdmm_type == "AR":
                # pose parameters in AR MultiLinear model do not contain jaw pose
                d_code = torch.cat([codedict['exp'], codedict['detail']], dim=1)
            else:
                d_code = torch.cat([codedict['pose'][:, 3:], codedict['exp'], codedict['detail']], dim=1)
            uv_z = self.D_detail(d_code)
            opdict['displacement_map'] = uv_z + self.fixed_uv_dis[None, None, ...]
            opdict['normals'] = util.vertex_normals(
                verts, self.render.faces.expand(batch_size, -1, -1),
            )

        return opdict            

    def visualize(self, visdict, size=224, dim=2):
        '''
        image range should be [0,1]
        dim: 2 for horizontal. 1 for vertical
        '''
        assert dim == 1 or dim == 2
        grids = {}
        for key in visdict:
            _, _, h, w = visdict[key].shape
            if dim == 2:
                new_h = size
                new_w = int(w * size / h)
            elif dim == 1:
                new_h = int(h * size / w)
                new_w = size
            grids[key] = torchvision.utils.make_grid(
                F.interpolate(visdict[key], [new_h, new_w]).detach().cpu(),
            )
        grid = torch.cat(list(grids.values()), dim)
        grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[..., [2, 1, 0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        return grid_image
    
    def save_obj(self, filename, opdict):
        '''
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        '''
        i = 0
        vertices = opdict['verts'][i].cpu().numpy()
        faces = self.render.faces[0].cpu().numpy()
        texture = util.tensor2image(opdict['uv_texture_gt'][i])
        uvcoords = self.render.raw_uvcoords[0].cpu().numpy()
        uvfaces = self.render.uvfaces[0].cpu().numpy()
        # save coarse mesh, with texture and normal map
        normal_map = util.tensor2image(opdict['uv_detail_normals'][i] * 0.5 + 0.5)
        util.write_obj(
            filename,
            vertices,
            faces, 
            texture=texture, 
            uvcoords=uvcoords, 
            uvfaces=uvfaces, 
            normal_map=normal_map,
        )
        # upsample mesh, save detailed mesh
        texture = texture[..., [2, 1, 0]]
        normals = opdict['normals'][i].cpu().numpy()
        displacement_map = opdict['displacement_map'][i].cpu().numpy().squeeze()
        dense_vertices, dense_colors, dense_faces = util.upsample_mesh(
            vertices, normals, displacement_map, texture, self.dense_template,
        )
        util.write_obj(
            filename.replace('.obj', '_detail.obj'), 
            dense_vertices, 
            dense_faces,
            colors=dense_colors,
            inverse_face_order=True,
        )

    def model_dict(self):
        return {
            'E_flame': self.E_flame.state_dict(),
            'E_detail': self.E_detail.state_dict(),
            'D_detail': self.D_detail.state_dict(),
        }
