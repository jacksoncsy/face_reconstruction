import cv2
import os.path as osp
import torch
import numpy as np

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, unique
from typing import Union, Optional, Dict, List

from .deca import DecaCoarse, DecaDetail, DecaSettings
from .deca_utils import (
    batch_orth_proj,
    bbox2point,
    compute_similarity_transform,
    compute_vertex_normals,
    parse_bbox_from_landmarks,
    transform_image_cv2,
    transform_to_image_space,
    transform_to_normalised_image_space,
    write_obj,
)
from .tdmm import FLAME, ARMultilinear, ARLinear, DetailSynthesiser
from .renderer import MeshRenderer


__all__ = ["DecaCoarsePredictor", "DecaDetailPredictor"]


@dataclass
class ModelConfig:
    weight_path: str
    settings: DecaSettings


@dataclass
class PredictorConfig:
    use_jit: bool


class DecaModelName(Enum):
    @classmethod
    def has_value(cls, value: str) -> bool:
        try:
            cls(value)
        except ValueError:
            return False
        return True

@unique
class DecaCoarseModelName(DecaModelName):
    FLAME_RES50_COARSE  = "flame_res50_coarse"
    FLAME_MBV2_COARSE   = "flame_mbv2_coarse"
    ARML_RES50_COARSE   = "arml_res50_coarse"
    ARML_MBV2_COARSE    = "arml_mbv2_coarse"
    ARL_RES50_COARSE    = "arl_res50_coarse"
    ARL_MBV2_COARSE     = "arl_mbv2_coarse"
    ARLV1_RES50_COARSE  = "arlv1_res50_coarse"
    ARLV1_MBV2_COARSE   = "arlv1_mbv2_coarse"

@unique
class DecaDetailModelName(DecaModelName):
    ARLV1_RES50_DETAIL  = "arlv1_res50_detail"


class DecaCoarsePredictor(object):
    def __init__(
        self, 
        device: Union[str, torch.device]="cuda:0",
        model_config: Optional[ModelConfig]=None,
        predictor_config: Optional[PredictorConfig]=None,
    ) -> None:
        self.device = device
        # all the settings for the network and the corresponding 3DMMs
        if model_config is None:
            model_config = DecaCoarsePredictor.create_model_config()
        self.model_config = model_config
        # record the type of 3DMMs
        self.tdmm_type = self.model_config.settings.tdmm_type.lower()
        
        # all the other settings for the predictor 
        if predictor_config is None:
            predictor_config = DecaCoarsePredictor.create_predictor_config()
        self.predictor_config = predictor_config
        
        # load network to predict parameters
        self.net = DecaCoarse(config=self.model_config.settings)
        self.net.load_state_dict(
            torch.load(self.model_config.weight_path, map_location=self.device)["state_dict"]
        )
        self.net.eval()
        
        # load 3DMM and other related assets
        self.tdmm = DecaCoarsePredictor.load_tdmm(self.model_config.settings)
        self.tdmm.eval()
        # record the trilist
        self.trilist = self.tdmm.get_trilist().copy()

        # load a mesh renderer
        self.mesh_renderer = MeshRenderer()
        self.mesh_renderer.eval()

        if self.predictor_config.use_jit:
            input_size = self.model_config.settings.input_size
            self.net = torch.jit.trace(
                self.net,
                torch.rand(1, 3, input_size, input_size),
            )
            self.tdmm = torch.jit.script(self.tdmm)

        self.net.to(self.device)
        self.tdmm.to(self.device)
        self.mesh_renderer.to(self.device)

    @staticmethod
    def create_model_config(name: str="arlv1_res50_coarse") -> ModelConfig:
        name = name.lower()
        assert DecaCoarseModelName.has_value(name), f"Unknown model name: {name}"
        method = DecaCoarseModelName(name)
        if method == DecaCoarseModelName.ARML_RES50_COARSE:
            return ModelConfig(
                weight_path=osp.join(osp.dirname(__file__), "weights/arml_res50_coarse.pth"),
                settings=DecaSettings(
                    tdmm_type="arml",
                    backbone="resnet50",
                    input_size=224,
                    coarse_parameters=OrderedDict(
                        {"shape": 72, "tex": 23, "exp": 52, "pose": 6, "cam": 3, "light": 27}
                    ),
                ),
            )
        elif method == DecaCoarseModelName.ARML_MBV2_COARSE:
            return ModelConfig(
                weight_path=osp.join(osp.dirname(__file__), "weights/arml_mbv2_coarse.pth"),
                settings=DecaSettings(
                    tdmm_type="arml",
                    backbone="mobilenetv2",
                    input_size=224,
                    coarse_parameters=OrderedDict(
                        {"shape": 72, "tex": 23, "exp": 52, "pose": 6, "cam": 3, "light": 27}
                    ),
                ),
            )
        elif method == DecaCoarseModelName.ARL_RES50_COARSE:
            return ModelConfig(
                weight_path=osp.join(osp.dirname(__file__), "weights/arl_res50_coarse.pth"),
                settings=DecaSettings(
                    tdmm_type="arl",
                    backbone="resnet50",
                    input_size=224,
                    coarse_parameters=OrderedDict(
                        {"shape": 33, "tex": 23, "exp": 52, "pose": 6, "cam": 3, "light": 27}
                    ),
                ),
            )
        elif method == DecaCoarseModelName.ARL_MBV2_COARSE:
            return ModelConfig(
                weight_path=osp.join(osp.dirname(__file__), "weights/??.pth"),
                settings=DecaSettings(
                    tdmm_type="arl",
                    backbone="mobilenetv2",
                    input_size=224,
                    coarse_parameters=OrderedDict(
                        {"shape": 33, "tex": 23, "exp": 52, "pose": 6, "cam": 3, "light": 27}
                    ),
                ),
            )
        elif method == DecaCoarseModelName.ARLV1_RES50_COARSE:
            return ModelConfig(
                weight_path=osp.join(osp.dirname(__file__), "weights/arlv1_res50_coarse.pth"),
                settings=DecaSettings(
                    tdmm_type="arlv1",
                    backbone="resnet50",
                    input_size=224,
                    coarse_parameters=OrderedDict(
                        {"shape": 33, "tex": 23, "exp": 52, "pose": 6, "cam": 3, "light": 27}
                    ),
                ),
            )
        elif method == DecaCoarseModelName.ARLV1_MBV2_COARSE:
            return ModelConfig(
                weight_path=osp.join(osp.dirname(__file__), "weights/??.pth"),
                settings=DecaSettings(
                    tdmm_type="arlv1",
                    backbone="mobilenetv2",
                    input_size=224,
                    coarse_parameters=OrderedDict(
                        {"shape": 33, "tex": 23, "exp": 52, "pose": 6, "cam": 3, "light": 27}
                    ),
                ),
            )             
        elif method == DecaCoarseModelName.FLAME_RES50_COARSE:
            return ModelConfig(
                weight_path=osp.join(osp.dirname(__file__), "weights/flame_res50_coarse.pth"),
                settings=DecaSettings(
                    tdmm_type="flame",
                    backbone="resnet50",
                    input_size=224,
                    coarse_parameters=OrderedDict(
                        {"shape": 100, "tex": 50, "exp": 50, "pose": 6, "cam": 3, "light": 27}
                    ),
                ),
            )
        elif method == DecaCoarseModelName.FLAME_MBV2_COARSE:
            return ModelConfig(
                weight_path=osp.join(osp.dirname(__file__), "weights/flame_mbv2_coarse.pth"),
                settings=DecaSettings(
                    tdmm_type="flame",
                    backbone="mobilenetv2",
                    input_size=224,
                    coarse_parameters=OrderedDict(
                        {"shape": 100, "tex": 50, "exp": 50, "pose": 6, "cam": 3, "light": 27}
                    ),
                ),
            )
        else:
            raise ValueError(f"Unknown model name: {name}")
    
    @staticmethod
    def create_predictor_config(use_jit: bool=True) -> PredictorConfig:
        return PredictorConfig(use_jit=use_jit)
    
    @staticmethod
    def load_tdmm(config: DecaSettings) -> Union[ARMultilinear, ARLinear, FLAME]:
        tdmm_type = config.tdmm_type.lower()
        if tdmm_type == "arml":
            tdmm = ARMultilinear(osp.join(osp.dirname(__file__), "assets/ar_multilinear"))
        elif tdmm_type == "arl":
            tdmm = ARLinear(osp.join(osp.dirname(__file__), "assets/ar_linear"))
        elif tdmm_type == "arlv1":
            tdmm = ARLinear(
                osp.join(osp.dirname(__file__), "assets/ar_linear_v1"),
                lmk_embedding_name="landmark_embedding_lms100.pkl",
            )
        elif tdmm_type == "flame":
            tdmm = FLAME(
                osp.join(osp.dirname(__file__), "assets/flame"),
                n_shape=config.coarse_parameters["shape"],
                n_exp=config.coarse_parameters["exp"],
            )
        else:
            raise ValueError(f"Unknown 3DMM type: {tdmm_type}")

        return tdmm

    @torch.no_grad()
    def __call__(
        self, image: np.ndarray, landmarks: np.ndarray, rgb: bool=True
    ) -> List[Dict]:
        if landmarks.size > 0:
            batch_size = landmarks.shape[0]
            h, w = image.shape[:2]
            # DECA expects RGB image as input
            if not rgb:
                image = image[..., ::-1]
            # convert to (bs, n_lmk, 2)
            if landmarks.ndim == 2:
                landmarks = landmarks[np.newaxis, ...]

            # Crop the faces
            bboxes = []
            batch_face = []
            batch_tform = []
            input_size = self.model_config.settings.input_size
            for lms in landmarks:
                bbox = parse_bbox_from_landmarks(lms)
                bboxes.append(bbox)
                src_size, src_center = bbox2point(bbox)
                # move the detected face to a standard frame
                tform = compute_similarity_transform(src_size, src_center, input_size)
                batch_tform.append(tform.params)
                crop_image = transform_image_cv2(image / 255., tform, input_size)
                batch_face.append(crop_image)

            # (bs, 4)
            bboxes = np.array(bboxes)
            # (bs, 3, 3)
            batch_tform = np.array(batch_tform)
            batch_tform = torch.from_numpy(batch_tform).float().to(self.device)
            # (bs, C, H, W)
            batch_face = np.array(batch_face).transpose((0, 3, 1, 2))
            batch_face = torch.from_numpy(batch_face).float().to(self.device)

            # Get parameters including 3DMMs, pose, light, camera etc.
            params = self.net(batch_face)

            # Parse those parameters according to the config
            params_dict = self.parse_parameters(params)

            # Clamp the expression parameters for certain 3DMMs
            if self.tdmm_type in ["arl", "arml"]:
                params_dict["exp"] = torch.clamp(params_dict["exp"], min=0.0, max=1.0)
            elif self.tdmm_type in ["arlv1"]:
                params_dict["exp"] = torch.sigmoid(params_dict["exp"])

            # Reconstruct using shape, expression and pose parameters
            # Results are in world coordinates, we will bring them to the original image space3
            # Also returns face poses (bs, 3) in radians with yaw-pitch-roll order
            vertices_world, landmarks2d_world, landmarks3d_world, face_poses = self.tdmm(
                shape_params=params_dict["shape"],
                expression_params=params_dict["exp"],
                pose_params=params_dict["pose"],
            )

            # Get projected vertices and landmarks in the crop image space
            landmarks2d = batch_orth_proj(landmarks2d_world, params_dict["cam"])[..., :2]
            landmarks2d[..., 1:] = -landmarks2d[..., 1:]
            
            landmarks3d = batch_orth_proj(landmarks3d_world, params_dict["cam"])
            landmarks3d[..., 1:] = -landmarks3d[..., 1:]
            
            vertices = batch_orth_proj(vertices_world, params_dict["cam"])
            vertices[..., 1:] = -vertices[..., 1:]

            # Recover to the original image space
            batch_inv_tform = torch.inverse(batch_tform).transpose(1,2).to(self.device)

            landmarks2d = transform_to_image_space(landmarks2d, batch_inv_tform, input_size)
            landmarks3d = transform_to_image_space(landmarks3d, batch_inv_tform, input_size)
            vertices = transform_to_image_space(vertices, batch_inv_tform, input_size)

            results = []
            for i in range(batch_size):
                results.append(
                    {
                        "bboxes": bboxes[i], # (4,)
                        "params_dict": {k:v[i].cpu().numpy() for k, v in params_dict.items()},
                        "vertices": vertices[i].cpu().numpy(), # (nv, 3)
                        "landmarks2d": landmarks2d[i].cpu().numpy(), # (n_lmk, 2)
                        "landmarks3d": landmarks3d[i].cpu().numpy(), # (n_lmk, 3)
                        "vertices_world": vertices_world[i].cpu().numpy(), # (nv, 3)
                        "landmarks3d_world": landmarks3d_world[i].cpu().numpy(), # (n_lmk, 3)
                        "face_poses": face_poses[i].cpu().numpy(), # (3,)
                        "inverse_transform": batch_inv_tform[i].cpu().numpy(), # (3,)
                    }
                )
            return results
        else:
            return []

    def parse_parameters(self, parameters: torch.Tensor) -> Dict:
        """
        args:
            parameters (bs, n_params): parameters predicted by deca
        """
        params_dict = {}
        curr_i = 0
        for k, v in self.model_config.settings.coarse_parameters.items():
            params_dict[k] = parameters[:, curr_i:curr_i+v]
            curr_i += v
        
        return params_dict

    def get_trilist(self) -> np.array:
        return self.trilist

    def render_shape_to_image(
        self,
        image: np.array,
        vertices: np.array,
        tri_faces: np.array,
        tform: np.array,
        cam: np.array,
    ) -> np.array:
        """
        args:
            image (bs, h, w, c): original image
            vertices (bs, nv, 3): vertices in world coordinates (not image coordinates!)
            tri_faces (bs, ntri, 3): triangles
            tform (bs, 3, 3): similarity transform from crop image back to the original
            cam: (bs, 3): orthographic projection from world space to image space
        """
        batch_size = vertices.shape[0]
        h, w = image.shape[1:3]

        vertices = torch.from_numpy(vertices).to(self.device)
        tri_faces = torch.from_numpy(tri_faces).long().to(self.device)
        tform = torch.from_numpy(tform).to(self.device)
        cam = torch.from_numpy(cam).to(self.device)

        image_tensor = torch.from_numpy(image.transpose((0, 3, 1, 2)) / 255.).float()
        batch_image = image_tensor.expand(batch_size, -1, -1, -1)
        batch_image = batch_image.to(self.device)

        # get normalised vertices with regard to original image resolution
        vertices_image = batch_orth_proj(vertices, cam)
        vertices_image[..., 1:] = -vertices_image[..., 1:]
        vertices_normalised = transform_to_normalised_image_space(
            vertices_image, tform, self.model_config.settings.input_size, (h, w)
        )

        rendered_images = self.mesh_renderer.render_shape(
            vertices, vertices_normalised, tri_faces, h, w, images=batch_image
        )
        # convert to uint8
        rendered_images = torch.clamp(rendered_images, min=0.0, max=1.0)
        # (bs, c, h, w) -> (bs, h, w, c)
        rendered_images = rendered_images.permute(0, 2, 3, 1).cpu().numpy()
        rendered_images = (255.0 * rendered_images).astype(np.uint8)

        return rendered_images


class DecaDetailPredictor(DecaCoarsePredictor):
    def __init__(
        self, 
        device: Union[str, torch.device]="cuda:0",
        model_config: Optional[ModelConfig]=None,
        predictor_config: Optional[PredictorConfig]=None,
    ) -> None:
        super(DecaDetailPredictor, self).__init__()
        self.device = device
        # all the settings for the network and the corresponding 3DMMs
        if model_config is None:
            model_config = DecaDetailPredictor.create_model_config()
        self.model_config = model_config
        # record the type of 3DMMs
        self.tdmm_type = self.model_config.settings.tdmm_type.lower()
        
        # all the other settings for the predictor 
        if predictor_config is None:
            predictor_config = DecaDetailPredictor.create_predictor_config()
        self.predictor_config = predictor_config
        
        # load coarse model 
        self.coarse_net = DecaCoarse(config=self.model_config.settings)
        self.coarse_net.load_state_dict(
            torch.load(self.model_config.weight_path, map_location=self.device)["state_dict_coarse"]
        )
        self.coarse_net.eval()

        # load detail model 
        self.detail_net = DecaDetail(config=self.model_config.settings)
        self.detail_net.load_state_dict(
            torch.load(self.model_config.weight_path, map_location=self.device)["state_dict_detail"]
        )
        self.detail_net.eval()
        
        # load 3DMM and other related assets
        self.tdmm = DecaDetailPredictor.load_tdmm(self.model_config.settings)
        self.tdmm.eval()
        # record the trilist
        self.trilist = self.tdmm.get_trilist().copy()

        # load synthesiser to get final displacement map and detail mesh
        self.detail_synthesiser = DecaDetailPredictor.load_detail_synthesiser(
            self.model_config.settings
        )
        self.detail_synthesiser.eval()

        # load a mesh renderer
        self.mesh_renderer = MeshRenderer()
        self.mesh_renderer.eval()

        if self.predictor_config.use_jit:
            input_size = self.model_config.settings.input_size
            self.coarse_net = torch.jit.trace(
                self.coarse_net, torch.rand(1, 3, input_size, input_size)
            )
            self.detail_net = torch.jit.script(self.detail_net)
            self.tdmm = torch.jit.script(self.tdmm)
            self.detail_synthesiser = torch.jit.script(self.detail_synthesiser)

        self.coarse_net.to(self.device)
        self.detail_net.to(self.device)
        self.tdmm.to(self.device)
        self.mesh_renderer.to(self.device)
        self.detail_synthesiser.to(self.device)

    @staticmethod
    def create_model_config(name: str="arlv1_res50_detail") -> ModelConfig:
        name = name.lower()
        assert DecaDetailModelName.has_value(name), f"Unknown model name: {name}"
        method = DecaDetailModelName(name)        
        if method == DecaDetailModelName.ARLV1_RES50_DETAIL:
            return ModelConfig(
                weight_path=osp.join(osp.dirname(__file__), "weights/arlv1_res50_detail_test.pth"),
                settings=DecaSettings(
                    tdmm_type="arlv1",
                    backbone="resnet50",
                    input_size=224,
                    coarse_parameters=OrderedDict(
                        {"shape": 33, "tex": 23, "exp": 52, "pose": 6, "cam": 3, "light": 27}
                    ),
                    detail_scale=10.0,
                    detail_parameters=OrderedDict(
                        {"exp": 52, "detail": 128}
                    ),
                ),
            )
        else:
            raise ValueError(f"Unknown model name: {name}")

    @staticmethod
    def load_detail_synthesiser(config: DecaSettings) -> DetailSynthesiser:
        tdmm_type = config.tdmm_type.lower()
        if tdmm_type in "arlv1":
            synthesiser = DetailSynthesiser(
                osp.join(osp.dirname(__file__), "assets/ar_linear_v1")
            )
        else:
            raise ValueError(f"Unknown 3DMM type for detail assets: {tdmm_type}")

        return synthesiser

    @torch.no_grad()
    def __call__(
        self, image: np.ndarray, landmarks: np.ndarray, rgb: bool=True
    ) -> List[Dict]:
        if landmarks.size > 0:
            batch_size = landmarks.shape[0]
            # DECA expects RGB image as input
            if not rgb:
                image = image[..., ::-1]
            # convert to (bs, n_lmk, 2)
            if landmarks.ndim == 2:
                landmarks = landmarks[np.newaxis, ...]

            # Crop the faces
            bboxes = []
            batch_face = []
            batch_tform = []
            input_size = self.model_config.settings.input_size
            for lms in landmarks:
                bbox = parse_bbox_from_landmarks(lms)
                bboxes.append(bbox)
                src_size, src_center = bbox2point(bbox)
                # move the detected face to a standard frame
                tform = compute_similarity_transform(src_size, src_center, input_size)
                batch_tform.append(tform.params)
                crop_image = transform_image_cv2(image / 255., tform, input_size)
                batch_face.append(crop_image)

            # (bs, 4)
            bboxes = np.array(bboxes)
            # (bs, 3, 3)
            batch_tform = np.array(batch_tform)
            batch_tform = torch.from_numpy(batch_tform).float().to(self.device)
            # (bs, C, H, W)
            batch_face = np.array(batch_face).transpose((0, 3, 1, 2))
            batch_face = torch.from_numpy(batch_face).float().to(self.device)

            # Get coarse parameters including 3DMMs, pose, light, camera.
            coarse_params = self.coarse_net(batch_face)

            # Parse coarse parameters according to the config
            params_dict = self.parse_parameters(coarse_params)

            # Clamp the expression parameters for certain 3DMMs
            if self.tdmm_type in ["arl", "arml"]:
                params_dict["exp"] = torch.clamp(params_dict["exp"], min=0.0, max=1.0)
            elif self.tdmm_type in ["arlv1"]:
                params_dict["exp"] = torch.sigmoid(params_dict["exp"])

            # Get detail parameters
            detail_params = self.detail_net(batch_face)
            params_dict["detail"] = detail_params
            uv_z = self.detail_net.decode(params_dict)
            displacement_map = self.detail_synthesiser(uv_z)

            # Reconstruct using shape, expression and pose parameters
            # Results are in world coordinates, we will bring them to the original image space3
            # Also returns face poses (bs, 3) in radians with yaw-pitch-roll order
            vertices_world, landmarks2d_world, landmarks3d_world, face_poses = self.tdmm(
                shape_params=params_dict["shape"],
                expression_params=params_dict["exp"],
                pose_params=params_dict["pose"],
            )

            # Get projected vertices and landmarks in the crop image space
            landmarks2d = batch_orth_proj(landmarks2d_world, params_dict["cam"])[..., :2]
            landmarks2d[..., 1:] = -landmarks2d[..., 1:]
            
            landmarks3d = batch_orth_proj(landmarks3d_world, params_dict["cam"])
            landmarks3d[..., 1:] = -landmarks3d[..., 1:]
            
            vertices = batch_orth_proj(vertices_world, params_dict["cam"])
            vertices[..., 1:] = -vertices[..., 1:]

            # Recover to the original image space
            batch_inv_tform = torch.inverse(batch_tform).transpose(1,2).to(self.device)

            landmarks2d = transform_to_image_space(landmarks2d, batch_inv_tform, input_size)
            landmarks3d = transform_to_image_space(landmarks3d, batch_inv_tform, input_size)
            vertices = transform_to_image_space(vertices, batch_inv_tform, input_size)

            # Get the detail mesh
            tri_faces = torch.from_numpy(self.trilist[None, ...]).to(self.device)
            tri_faces = tri_faces.expand(batch_size, -1, -1)
            detail_results = self.detail_synthesiser.upsample_mesh(
                vertices_world, tri_faces, displacement_map
            )
            dense_vertices_world = detail_results["dense_vertices"]
            dense_faces = detail_results["dense_faces"]

            results = []
            for i in range(batch_size):
                results.append(
                    {
                        "bboxes": bboxes[i], # (4,)
                        "params_dict": {k:v[i].cpu().numpy() for k, v in params_dict.items()},
                        "vertices": vertices[i].cpu().numpy(), # (nv, 3)
                        "landmarks2d": landmarks2d[i].cpu().numpy(), # (n_lmk, 2)
                        "landmarks3d": landmarks3d[i].cpu().numpy(), # (n_lmk, 3)
                        "vertices_world": vertices_world[i].cpu().numpy(), # (nv, 3)
                        "landmarks3d_world": landmarks3d_world[i].cpu().numpy(), # (n_lmk, 3)
                        "face_poses": face_poses[i].cpu().numpy(), # (3,)
                        "inverse_transform": batch_inv_tform[i].cpu().numpy(), # (3,)
                        "uv_z": uv_z[i].cpu().numpy().transpose((1, 2, 0)), # (h, w, c)
                        "dense_vertices_world": dense_vertices_world[i].cpu().numpy(), # (nv_dense, 3)
                        "dense_faces": dense_faces[i].cpu().numpy(), # (ntri_dense, 3)
                    }
                )
            return results
        else:
            return []
