import numpy as np
import torch

from skimage.transform import estimate_transform, warp, _geometric
from typing import Union


def batch_orth_proj(X: torch.tensor, camera: torch.tensor) -> torch.tensor:
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:, :, 2:]], 2)
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn


def bbox2point(bbox):
    ''' take only the bbox from landmarks
    '''
    left, top, right, bottom = bbox
    size = 1.1 * (right - left + bottom - top) / 2.
    center = np.array([
        right - (right - left) / 2.,
        bottom - (bottom - top) / 2.,
    ])
    return size, center


def check_light(light_params: np.array, threshold: float=15.) -> bool:
    """
    light_params: (27,), lighting parameters
    """
    # abnormal lighting prediction usually results in extremely large values
    return False if np.mean(light_params[:3]) >= threshold else True
    

def check_2d_landmarks(
    gt_landmarks: np.array, pred_landmarks: np.array, threshold: float=0.2,
) -> bool:
    """
    gt_landmarks: (n_lmk, 2), ground truth 2D landmarks
    pred_landmarks: (n_lmk, 2), predicted 2D landmarks
    """
    # compute normalise RMSE
    face_diagonal = np.linalg.norm(np.max(gt_landmarks, axis=0) - np.min(gt_landmarks, axis=0))
    rmse = np.mean(np.linalg.norm(gt_landmarks - pred_landmarks, axis=1))
    return False if rmse / face_diagonal >= threshold else True


def compute_similarity_transform(
    src_size: Union[float, int],
    src_center: np.array,
    crop_size: int,
    scale: float=1.25,
) -> _geometric.GeometricTransform:
    size = int(src_size * scale)
    src_pts = np.array([
        [src_center[0] - size / 2, src_center[1] - size / 2],
        [src_center[0] - size / 2, src_center[1] + size / 2],
        [src_center[0] + size / 2, src_center[1] - size / 2],
    ])

    dst_pts = np.array([
        [0, 0], 
        [0, crop_size - 1],
        [crop_size - 1, 0],
    ])

    tform = estimate_transform('similarity', src_pts, dst_pts)
    return tform


def parse_bbox_from_landmarks(landmarks: np.array):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])
    bbox = [left, top, right, bottom]
    return bbox


def transform_image(
    image: np.array, tform: _geometric.GeometricTransform, crop_size: int,
) -> np.array:
    return warp(image, tform.inverse, output_shape=(crop_size, crop_size))


def transform_to_image_space(
    points: torch.tensor, tform: torch.tensor, crop_size: int,
) -> torch.tensor:
    last_dim = points.shape[-1]
    assert last_dim == 2 or last_dim == 3

    points_2d = points[..., :2]
        
    #input points must use original range
    points_2d = (points_2d * 0.5 + 0.5) * crop_size

    batch_size, n_points, _ = points.shape
    trans_points_2d = torch.bmm(
        torch.cat([points_2d, torch.ones([batch_size, n_points, 1], device=points.device, dtype=points.dtype)], dim=-1), 
        tform,
    )

    if last_dim == 2:
        trans_points = trans_points_2d[..., :2]
    else:
        # guess the z as well
        scales = (tform[:, 0, 0] + tform[:, 1, 1]) / 2
        z_coords = -scales[:, None, None] * crop_size * (points[..., [2]] * 0.5 + 0.5)
        trans_points = torch.cat([trans_points_2d[..., :2], z_coords], dim=-1)
    
    return trans_points

