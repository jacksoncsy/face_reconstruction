import os
import torch
import numpy as np

from collections import OrderedDict
from types import SimpleNamespace
from typing import Union, Optional, Dict

from .deca import DecaCoarse
from .deca_utils import (
    batch_orth_proj,
    bbox2point,
    compute_similarity_transform,
    parse_bbox_from_landmarks,
    transform_image,
    transform_to_image_space,
)
from .tdmm import FLAME, ARMultilinear


__all__ = ["DecaCoarsePredictor"]


class DecaCoarsePredictor(object):
    def __init__(
        self, 
        device: Union[str, torch.device]="cuda:0",
        model_config: Optional[SimpleNamespace]=None,
        predictor_config: Optional[SimpleNamespace]=None,
    ) -> None:
        self.device = device
        # all the settings for the network and the corresponding 3DMMs
        if model_config is None:
            model_config = DecaCoarsePredictor.create_model_config()
        # all the other settings for the predictor 
        if predictor_config is None:
            predictor_config = DecaCoarsePredictor.create_predictor_config()
        
        self.config = SimpleNamespace(
            **model_config.settings.__dict__,
            **predictor_config.__dict__,
        )

        # load network to predict parameters
        self.net = DecaCoarse(config=self.config).to(self.device)
        self.net.load_state_dict(torch.load(model_config.weight_path, map_location=self.device))
        self.net.eval()
        
        # load 3DMM and other related assets
        self.tdmm = DecaCoarsePredictor.load_tdmm(self.config).to(self.device)
        self.tdmm.eval()

        if self.config.use_jit:
            self.net = torch.jit.trace(
                self.net,
                torch.rand(1, 3, self.config.input_size, self.config.input_size).to(self.device),
            )
            # TODO: does tdmm need to be traceable?
            self.tdmm = torch.jit.trace(
                self.tdmm,
                (
                    torch.rand(1, self.config.coarse_parameters["shape"]).to(self.device),
                    torch.rand(1, self.config.coarse_parameters["exp"]).to(self.device),
                    torch.rand(1, self.config.coarse_parameters["pose"]).to(self.device),
                )
            )

    @staticmethod
    def create_model_config(name: str="ar_res50_coarse") -> SimpleNamespace:
        name = name.lower()
        if name == "ar_res50_coarse":
            return SimpleNamespace(
                weight_path=os.path.join(os.path.dirname(__file__), "weights/ar_res50_coarse.pth"),
                settings=SimpleNamespace(
                    tdmm_type="ar",
                    backbone="resnet50",
                    input_size=224,
                    coarse_parameters=OrderedDict(
                        {"shape": 72, "tex": 23, "exp": 52, "pose": 6, "cam": 3, "light": 27}
                    ),
                ),
            )
        elif name == "ar_mbv2_coarse":
            return SimpleNamespace(
                weight_path=os.path.join(os.path.dirname(__file__), "weights/ar_mbv2_coarse.pth"),
                settings=SimpleNamespace(
                    tdmm_type="ar",
                    backbone="mobilenetv2",
                    input_size=224,
                    coarse_parameters=OrderedDict(
                        {"shape": 72, "tex": 23, "exp": 52, "pose": 6, "cam": 3, "light": 27}
                    ),
                ),
            )
        elif name == "flame_res50_coarse":
            return SimpleNamespace(
                weight_path=os.path.join(os.path.dirname(__file__), "weights/flame_res50_coarse.pth"),
                settings=SimpleNamespace(
                    tdmm_type="flame",
                    backbone="resnet50",
                    input_size=224,
                    coarse_parameters=OrderedDict(
                        {"shape": 100, "tex": 50, "exp": 50, "pose": 6, "cam": 3, "light": 27}
                    ),
                ),
            )
        elif name == "flame_mbv2_coarse":
            return SimpleNamespace(
                weight_path=os.path.join(os.path.dirname(__file__), "weights/flame_mbv2_coarse.pth"),
                settings=SimpleNamespace(
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
    def create_predictor_config(use_jit: bool=True) -> SimpleNamespace:
        return SimpleNamespace(use_jit=use_jit)
    
    @staticmethod
    def load_tdmm(config: SimpleNamespace) -> Union[ARMultilinear, FLAME]:
        tdmm_type = config.tdmm_type.lower()
        if tdmm_type == "ar":
            tdmm = ARMultilinear(os.path.join(os.path.dirname(__file__), "assets/ar_multilinear"))
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
    ) -> Union[Dict, None]:
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
            for lms in landmarks:
                bbox = parse_bbox_from_landmarks(lms)
                bboxes.append(bbox)
                src_size, src_center = bbox2point(bbox)
                # move the detected face to a standard frame
                tform = compute_similarity_transform(src_size, src_center, self.config.input_size)
                batch_tform.append(tform.params)
                crop_image = transform_image(image / 255., tform, self.config.input_size)
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
            h, w = image.shape[:2]
            batch_inv_tform = torch.inverse(batch_tform).transpose(1,2).to(self.device)

            landmarks2d = transform_to_image_space(landmarks2d, batch_inv_tform, self.config.input_size)
            landmarks3d = transform_to_image_space(landmarks3d, batch_inv_tform, self.config.input_size)
            vertices = transform_to_image_space(vertices, batch_inv_tform, self.config.input_size)
        
            return {
                "bboxes": bboxes, # (bs, 4)
                "params_dict": {k:v.cpu().numpy() for k, v in params_dict.items()},
                "vertices": vertices.cpu().numpy(), # (bs, nv, 3)
                "landmarks2d": landmarks2d.cpu().numpy(), # (bs, n_lmk, 2)
                "landmarks3d": landmarks3d.cpu().numpy(), # (bs, n_lmk, 3)
                "vertices_world": vertices_world.cpu().numpy(), # (bs, nv, 3)
                "landmarks3d_world": landmarks3d_world.cpu().numpy(), # (bs, n_lmk, 3)
                "face_poses": face_poses.cpu().numpy(), # (bs, 3)
            }
        else:
            return None

    def parse_parameters(self, parameters: torch.tensor) -> Dict:
        """
        parameters: (bs, n_params), parameters predicted by deca
        """
        params_dict = {}
        curr_i = 0
        for k, v in self.config.coarse_parameters.items():
            params_dict[k] = parameters[:, curr_i:curr_i+v]
            curr_i += v
        
        return params_dict

    def get_trilist(self) -> np.array:
        return self.tdmm.trilist
