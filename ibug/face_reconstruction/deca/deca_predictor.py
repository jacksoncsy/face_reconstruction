import os
import torch
import numpy as np

from collections import OrderedDict
from dataclasses import dataclass
from typing import Union, Optional, Dict, List

from .deca import DecaCoarse, DecaSettings
from .deca_utils import (
    batch_orth_proj,
    bbox2point,
    compute_similarity_transform,
    parse_bbox_from_landmarks,
    transform_image_cv2,
    transform_to_image_space,
)
from .tdmm import FLAME, ARMultilinear, ARLinear


__all__ = ["DecaCoarsePredictor"]


@dataclass
class ModelConfig:
    weight_path: str
    settings: DecaSettings


@dataclass
class PredictorConfig:
    use_jit: bool


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
        # record the type of 3DMMs
        self.tdmm_type = self.model_config.settings.tdmm_type

        if self.predictor_config.use_jit:
            input_size = self.model_config.settings.input_size
            self.net = torch.jit.trace(
                self.net,
                torch.rand(1, 3, input_size, input_size),
            )
            self.tdmm = torch.jit.script(self.tdmm)

        self.net.to(self.device)
        self.tdmm.to(self.device)

    @staticmethod
    def create_model_config(name: str="arml_res50_coarse") -> ModelConfig:
        name = name.lower()
        if name == "arml_res50_coarse":
            return ModelConfig(
                weight_path=os.path.join(os.path.dirname(__file__), "weights/arml_res50_coarse.pth"),
                settings=DecaSettings(
                    tdmm_type="arml",
                    backbone="resnet50",
                    input_size=224,
                    coarse_parameters=OrderedDict(
                        {"shape": 72, "tex": 23, "exp": 52, "pose": 6, "cam": 3, "light": 27}
                    ),
                ),
            )
        elif name == "arml_mbv2_coarse":
            return ModelConfig(
                weight_path=os.path.join(os.path.dirname(__file__), "weights/arml_mbv2_coarse.pth"),
                settings=DecaSettings(
                    tdmm_type="arml",
                    backbone="mobilenetv2",
                    input_size=224,
                    coarse_parameters=OrderedDict(
                        {"shape": 72, "tex": 23, "exp": 52, "pose": 6, "cam": 3, "light": 27}
                    ),
                ),
            )
        elif name == "arl_res50_coarse":
            return ModelConfig(
                weight_path=os.path.join(os.path.dirname(__file__), "weights/arl_res50_coarse.pth"),
                settings=DecaSettings(
                    tdmm_type="arl",
                    backbone="resnet50",
                    input_size=224,
                    coarse_parameters=OrderedDict(
                        {"shape": 33, "tex": 23, "exp": 52, "pose": 6, "cam": 3, "light": 27}
                    ),
                ),
            )
        elif name == "arl_mbv2_coarse":
            return ModelConfig(
                weight_path=os.path.join(os.path.dirname(__file__), "weights/??.pth"),
                settings=DecaSettings(
                    tdmm_type="arl",
                    backbone="mobilenetv2",
                    input_size=224,
                    coarse_parameters=OrderedDict(
                        {"shape": 33, "tex": 23, "exp": 52, "pose": 6, "cam": 3, "light": 27}
                    ),
                ),
            )            
        elif name == "flame_res50_coarse":
            return ModelConfig(
                weight_path=os.path.join(os.path.dirname(__file__), "weights/flame_res50_coarse.pth"),
                settings=DecaSettings(
                    tdmm_type="flame",
                    backbone="resnet50",
                    input_size=224,
                    coarse_parameters=OrderedDict(
                        {"shape": 100, "tex": 50, "exp": 50, "pose": 6, "cam": 3, "light": 27}
                    ),
                ),
            )
        elif name == "flame_mbv2_coarse":
            return ModelConfig(
                weight_path=os.path.join(os.path.dirname(__file__), "weights/flame_mbv2_coarse.pth"),
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
            tdmm = ARMultilinear(os.path.join(os.path.dirname(__file__), "assets/ar_multilinear"))
        elif tdmm_type == "arl":
            tdmm = ARLinear(os.path.join(os.path.dirname(__file__), "assets/ar_linear"))
        elif tdmm_type == "flame":
            tdmm = FLAME(
                os.path.join(os.path.dirname(__file__), "assets/flame"),
                n_shape=config.coarse_parameters["shape"],
                n_exp=config.coarse_parameters["exp"],
            )
        else:
            raise ValueError(f"Unknown 3DMM type: {tdmm_type}")

        return tdmm

    @torch.no_grad()
    def __call__(
        self, image: np.ndarray, landmarks: np.ndarray, rgb: bool=True,
    ) -> List[Dict]:
        if landmarks.size > 0:
            if rgb:
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

            batch_size = landmarks.shape[0]
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
                    }
                )
            return results
        else:
            return []

    def parse_parameters(self, parameters: torch.tensor) -> Dict:
        """
        parameters: (bs, n_params), parameters predicted by deca
        """
        params_dict = {}
        curr_i = 0
        for k, v in self.model_config.settings.coarse_parameters.items():
            params_dict[k] = parameters[:, curr_i:curr_i+v]
            curr_i += v
        
        return params_dict

    def get_trilist(self) -> np.array:
        return self.tdmm.trilist
