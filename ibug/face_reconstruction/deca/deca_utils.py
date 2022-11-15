import cv2
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F

from skimage.transform import estimate_transform, warp, _geometric
from typing import Tuple, Union


def batch_orth_proj(X: torch.Tensor, camera: torch.Tensor) -> torch.Tensor:
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


def check_light(light_params: np.ndarray, threshold: float=15.) -> bool:
    """
    light_params: (27,), lighting parameters
    """
    # abnormal lighting prediction usually results in extremely large values
    return False if np.mean(light_params[:3]) >= threshold else True
    

def check_2d_landmarks(
    gt_landmarks: np.ndarray, pred_landmarks: np.ndarray, threshold: float=0.2,
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


def compute_face_vertices(vertices: torch.Tensor, faces: torch.Tensor):
    """
    args:
        vertices: (bs, nv, 3)
        faces: (bs, ntri, 3)
    return:
        vertices per triangle (bs, ntri, 3, 3)
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


def compute_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor):
    """
    args:
        vertices: (bs, nv, 3)
        faces: (bs, nv, 3)
    return:
        vertex_normals: (bs, nv, 3)
    """
    assert vertices.ndim == 3 and faces.ndim == 3
    assert vertices.shape[0] == faces.shape[0] and \
        vertices.shape[2] == 3 and \
        faces.shape[2] == 3
    bs, nv = vertices.shape[:2]
    device = vertices.device
    
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None] # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.reshape(-1, 3)
    vertices_faces = vertices_faces.reshape(-1, 3, 3)

    vertex_normals = torch.zeros(bs * nv, 3).to(device)
    vertex_normals.index_add_(
        0, faces[:, 1].long(), 
        torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]),
    )
    vertex_normals.index_add_(
        0, faces[:, 2].long(), 
        torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]),
    )
    vertex_normals.index_add_(
        0, faces[:, 0].long(),
        torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]),
    )

    vertex_normals = F.normalize(vertex_normals, eps=1e-6, dim=1)
    vertex_normals = vertex_normals.reshape((bs, nv, 3))
    return vertex_normals
    

def parse_bbox_from_landmarks(landmarks: np.ndarray):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])
    bbox = [left, top, right, bottom]
    return bbox


def transform_image(
    image: np.ndarray, tform: _geometric.GeometricTransform, crop_size: int,
) -> np.ndarray:
    return warp(image, tform.inverse, output_shape=(crop_size, crop_size))


def transform_image_cv2(
    image: np.ndarray, tform: _geometric.GeometricTransform, crop_size: int,
) -> np.ndarray:
    return cv2.warpAffine(
        image,
        tform.params[:2],
        (crop_size, crop_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def transform_to_image_space(
    points: torch.Tensor, tform: torch.Tensor, crop_size: int,
) -> torch.Tensor:
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


def transform_to_normalised_image_space(
    points: torch.Tensor, tform: torch.Tensor, crop_size: int, image_size: Tuple[int, int]
) -> torch.Tensor:
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

    # normalise the coords to [-1, 1]
    trans_points_2d[..., 0] = trans_points_2d[..., 0] / image_size[1] * 2 - 1
    trans_points_2d[..., 1] = trans_points_2d[..., 1] / image_size[0] * 2 - 1
    if last_dim == 2:
        trans_points = trans_points_2d[..., :2]
    else:
        trans_points = torch.cat([trans_points_2d[..., :2], points[..., 2:]], dim=-1)
    
    return trans_points


def write_obj(
    save_path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: Union[np.ndarray, None] = None,
    texture: Union[np.ndarray, None] = None,
    uvcoords: Union[np.ndarray, None] = None,
    uvfaces: Union[np.ndarray, None] = None,
    normal_map: Union[np.ndarray, None] = None,
    inverse_face_order: bool = False,
):
    """ Save 3D face model with texture. 
    args:
        save_path: full target filepath for the obj
        vertices: (nv, 3)
        colors: (nv, 3)
        faces: (ntri, 3)
        texture: (uv_size, uv_size, 3)
        uvcoords: (nv, 2), max_value <= 1
        uvfaces: (ntri, 3)
        normal_map: (uv_size, uv_size, c)
    """
    if osp.splitext(save_path)[-1] != ".obj":
        save_path = save_path + ".obj"
    mtl_name = save_path.replace(".obj", ".mtl")
    texture_name = save_path.replace(".obj", ".png")
    material_name = "FaceTexture"

    faces = faces.copy()
    # mesh lab start with 1, python/c++ start from 0
    faces += 1
    if inverse_face_order:
        faces = faces[:, [2, 1, 0]]
        if uvfaces is not None:
            uvfaces = uvfaces[:, [2, 1, 0]]

    # write obj
    with open(save_path, "w") as f:
        # first line: write mtlib(material library)
        if texture is not None:
            f.write("mtllib %s\n\n" % osp.basename(mtl_name))

        # write vertices
        if colors is None:
            for i in range(vertices.shape[0]):
                f.write(
                    "v {} {} {}\n".format(
                        vertices[i, 0], vertices[i, 1], vertices[i, 2],
                    )
                )
        else:
            for i in range(vertices.shape[0]):
                f.write(
                    "v {} {} {} {} {} {}\n".format(
                        vertices[i, 0], vertices[i, 1], vertices[i, 2],
                        colors[i, 0], colors[i, 1], colors[i, 2],
                    )
                )

        # write uv coords
        if texture is None:
            for i in range(faces.shape[0]):
                f.write("f {} {} {}\n".format(faces[i, 2], faces[i, 1], faces[i, 0]))
        else:
            for i in range(uvcoords.shape[0]):
                f.write("vt {} {}\n".format(uvcoords[i,0], uvcoords[i,1]))
            f.write("usemtl %s\n" % material_name)
            # write f: ver ind/ uv ind
            uvfaces = uvfaces + 1
            for i in range(faces.shape[0]):
                f.write(
                    "f {}/{} {}/{} {}/{}\n".format(
                        faces[i, 0], uvfaces[i, 0],
                        faces[i, 1], uvfaces[i, 1],
                        faces[i, 2], uvfaces[i, 2],
                    )
                )
            # write mtl
            with open(mtl_name, "w") as f:
                f.write("newmtl %s\n" % material_name)
                s = "map_Kd {}\n".format(osp.basename(texture_name)) # map to image
                f.write(s)

                if normal_map is not None:
                    name, _ = osp.splitext(save_path)
                    normal_name = f"{name}_normals.png"
                    f.write(f"disp {normal_name}")
                    cv2.imwrite(normal_name, normal_map)

            cv2.imwrite(texture_name, texture)
