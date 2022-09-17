import os
import cv2
import torch
import numpy as np

from collections import OrderedDict
from skimage.transform._geometric import GeometricTransform
from types import SimpleNamespace
from typing import Union, Optional, List, Dict, Tuple

from .deca import DecaCoarse
from .deca_utils import (
    parse_bbox_from_landmarks,
    bbox2point,
    compute_similarity_transform,
    transform_image,
)
from .tdmm import FLAME, ARMultilinear


__all__ = ['DecaCoarsePredictor']


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
                    backbone='resnet50',
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
                    backbone='mobilenetv2',
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
                    backbone='resnet50',
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
                    backbone='mobilenetv2',
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
        self, 
        image: np.ndarray,
        landmarks: np.ndarray,
        rgb: bool=True,
    ) -> Tuple[np.ndarray, List[GeometricTransform]]:
        if landmarks.size > 0:
            if rgb:
                image = image[..., ::-1]
            # convert to (bs, n_lmk, 2)
            if landmarks.ndim == 2:
                landmarks = landmarks[np.newaxis, ...]

            # Crop the faces
            batch_face = []
            batch_tform = []
            for lms in landmarks:
                src_size, src_center = bbox2point(parse_bbox_from_landmarks(lms))
                # move the detected face to a standard frame
                tform = compute_similarity_transform(src_size, src_center, self.config.input_size)
                crop_image = transform_image(image / 255., tform, self.config.input_size)
                batch_face.append(crop_image)
                batch_tform.append(tform)
            # (bs, C, H, W)   
            batch_face = np.array(batch_face).transpose((0, 3, 1, 2))
            batch_face = torch.from_numpy(batch_face).float().to(self.device)

            # Get 3DMM parameters
            params = self.net(batch_face).cpu().numpy()

            return params, batch_tform
        else:
            return np.empty(shape=(0, self.net.output_size), dtype=np.float32), []

    @staticmethod
    def decode(tdmm_params: np.ndarray, pose_pref: int = 0) -> Union[Dict, List[Dict]]:
        if tdmm_params.size > 0:
            if tdmm_params.ndim > 1:
                return [DECAPredictor.decode(x) for x in tdmm_params]
            else:
                roi_box = tdmm_params[:4]
                params = tdmm_params[4:]
                vertex, pts68, f_rot, tr = reconstruct_from_3dmm(params)
                camera_transform = {'fR': f_rot, 'T': tr}
                pitch, yaw, roll, t3d, f = parse_param_pose(params, pose_pref)
                face_pose = {'pitch': pitch, 'yaw': yaw, 'roll': roll, 't3d': t3d, 'f': f}
                return {'roi_box': roi_box, 'params': params, 'vertex': vertex, 'pts68': pts68,
                        'face_pose': face_pose, 'camera_transform': camera_transform}
        else:
            return []
