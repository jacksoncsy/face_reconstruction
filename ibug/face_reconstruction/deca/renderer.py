# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .deca_utils import compute_face_vertices, compute_vertex_normals
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes
from typing import Union


class Pytorch3dRasterizer(nn.Module):
    """  Borrowed from https://github.com/facebookresearch/pytorch3d
    """
    def __init__(self):
        """
        use fixed raster_settings for rendering faces
        """
        super(Pytorch3dRasterizer, self).__init__()
        self.raster_settings = {
            "blur_radius": 0.0,
            "faces_per_pixel": 1,
            "bin_size": None,
            "max_faces_per_bin": None,
            "perspective_correct": False,
        }

    def forward(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        h: int,
        w: int,
        attributes: Union[torch.Tensor, None] = None,
    ):
        """
        Notice:
            vertices (bs, nv, 3) should be in image space, and are normalized
        """
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]
        
        if h > w:
            fixed_vertices[..., 1] = fixed_vertices[..., 1] * h / w
        else:
            fixed_vertices[..., 0] = fixed_vertices[..., 0] * w / h
        
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        pix_to_face, _, bary_coords, _ = rasterize_meshes(
            meshes_screen,
            image_size=[h, w],
            blur_radius=self.raster_settings["blur_radius"],
            faces_per_pixel=self.raster_settings["faces_per_pixel"],
            bin_size=self.raster_settings["bin_size"],
            max_faces_per_bin=self.raster_settings["max_faces_per_bin"],
            perspective_correct=self.raster_settings["perspective_correct"],
        )
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(
            attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1],
        )
        N, H, W, K, _ = bary_coords.shape
        mask = (pix_to_face == -1)
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        # Replace masked values in output.
        pixel_vals[mask] = 0
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None]], dim=1)
        return pixel_vals


class MeshRenderer(nn.Module):
    def __init__(self):
        super(MeshRenderer, self).__init__()
        # rasterizer for ordinary image space
        self.rasterizer = Pytorch3dRasterizer()

    def render_shape(
        self,
        vertices_world: torch.Tensor,
        vertices_image_normalised: torch.Tensor,
        tri_faces: torch.Tensor,
        h: int,
        w: int,
        images: Union[torch.Tensor, None] = None,
    ):
        """Rendering shape with detail normal map
        args:
            vertices_world (bs, nv, 3)
            vertices_image_normalised (bs, nv, 3)
            tri_faces (bs, ntri, 3)
            h: frame height
            w: frame width
            (Optional) images (bs, c, h, w): original images
        """
        assert vertices_image_normalised.ndim == 3 and \
                vertices_world.ndim == 3 and \
                tri_faces.ndim == 3 

        batch_size = vertices_world.shape[0]
        ntri = tri_faces.shape[1]
        device = vertices_world.device

        # set lighting
        light_positions = torch.tensor(
            [[-1, 1, 1], [1, 1, 1], [-1, -1, 1], [1, -1, 1], [0, 0, 1]]
        )[None, ...].expand(batch_size, -1, -1).float()

        light_intensities = 1.7 * torch.ones_like(light_positions).float()
        lights = torch.cat((light_positions, light_intensities), dim=2).to(device)

        transformed_vertices = vertices_image_normalised.clone()
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10

        # Attributes
        face_vertices = compute_face_vertices(vertices_world, tri_faces)
        normals = compute_vertex_normals(vertices_world, tri_faces)
        face_normals = compute_face_vertices(normals, tri_faces)
        transformed_normals = compute_vertex_normals(transformed_vertices, tri_faces)
        transformed_face_normals = compute_face_vertices(transformed_normals, tri_faces)
        
        face_colors = (180. / 255) * torch.ones((batch_size, ntri, 3, 3)).to(device)        
        attributes = torch.cat(
            [face_colors, transformed_face_normals, face_vertices, face_normals], dim=-1
        )
        # rasterize
        rendering = self.rasterizer(transformed_vertices, tri_faces, h, w, attributes)
        rendering = rendering.detach()

        alpha_images = rendering[:, [-1]]
        # albedo
        albedo_images = rendering[:, :3]
        # mask
        transformed_normal_map = rendering[:, 3:6]
        pos_mask = (transformed_normal_map[:, 2:] < 0.15).float()
        # shading
        normal_images = rendering[:, 9:12]
        
        shading = self.add_directionlight(
            normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]), lights
        )
        shading_images = shading.reshape([batch_size, h, w, 3]).permute(0, 3, 1, 2).contiguous()
        shaded_images = albedo_images * shading_images

        alpha_images = alpha_images * pos_mask

        if images is None:
            shape_images = shaded_images * alpha_images + \
                torch.zeros_like(shaded_images).to(device) * (1 - alpha_images)
        else:
            shape_images = shaded_images * alpha_images + images * (1 - alpha_images)
        
        return shape_images

    def add_directionlight(self, normals, lights):
        """
        args:
            normals: (bz, nv, 3)
            lights: (bz, nlight, 6)
        return:
            shading: (bz, nv, 3)
        """
        light_direction = lights[..., :3]
        light_intensities = lights[..., 3:]
        directions_to_lights = F.normalize(
            light_direction[:, :, None, :].expand(-1, -1, normals.shape[1], -1), dim=3
        )
        normals_dot_lights = torch.clamp(
            (normals[:, None, :, :] * directions_to_lights).sum(dim=3), min=0., max=1.
        )
        shading = normals_dot_lights[:, :, :, None] * light_intensities[:, :, None, :]
        return shading.mean(dim=1)