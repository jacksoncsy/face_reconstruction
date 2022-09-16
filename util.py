import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
import torchvision

from scipy.ndimage import morphology


PI = torch.Tensor([3.14159265358979323846])
LINES_KPTS68 = [
    list(range(17)),
    list(range(17, 22)),
    list(range(22, 27)),
    list(range(27, 31)),
    list(range(31, 36)),
    list(range(36, 42)) + [36],
    list(range(42, 48)) + [42],
    list(range(48, 60)) + [48],
    list(range(60, 68)) + [60],
]


def upsample_mesh(vertices, normals, displacement_map, texture_map, dense_template):
    ''' Credit to Timo
    upsampling coarse mesh (with displacment map)
        vertices: vertices of coarse mesh, [nv, 3]
        normals: vertex normals, [nv, 3]
        texture_map: texture map, [256, 256, 3]
        displacement_map: displacment map, [256, 256]
        dense_template: 
    Returns: 
        dense_vertices: upsampled vertices with details, [number of dense vertices, 3]
        dense_colors: vertex color, [number of dense vertices, 3]
        dense_faces: [number of dense faces, 3]
    '''
    dense_faces = dense_template['f']
    x_coords = dense_template['x_coords']
    y_coords = dense_template['y_coords']
    valid_pixel_ids = dense_template['valid_pixel_ids']
    valid_pixel_3d_faces = dense_template['valid_pixel_3d_faces']
    valid_pixel_b_coords = dense_template['valid_pixel_b_coords']

    pixel_3d_points = \
        vertices[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
        vertices[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
        vertices[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]
    pixel_3d_normals = \
        normals[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
        normals[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
        normals[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]
    pixel_3d_normals = pixel_3d_normals / np.linalg.norm(pixel_3d_normals, axis=-1)[:, np.newaxis]
    displacements = displacement_map[y_coords[valid_pixel_ids].astype(int), x_coords[valid_pixel_ids].astype(int)]
    dense_colors = texture_map[y_coords[valid_pixel_ids].astype(int), x_coords[valid_pixel_ids].astype(int)]
    offsets = np.einsum('i,ij->ij', displacements, pixel_3d_normals)
    dense_vertices = pixel_3d_points + offsets
    return dense_vertices, dense_colors, dense_faces


def upsample_mesh_without_texture(vertices, normals, displacement_map, dense_template):
    ''' 
    upsampling coarse mesh (with displacment map)
        vertices: vertices of coarse mesh, [nv, 3]
        normals: vertex normals, [nv, 3]
        displacement_map: displacment map, [256, 256]
        dense_template:
    Returns:
        dense_vertices: upsampled vertices with details, [number of dense vertices, 3]
        dense_faces: [number of dense faces, 3]
    '''
    dense_faces = dense_template['f']
    x_coords = dense_template['x_coords']
    y_coords = dense_template['y_coords']
    valid_pixel_ids = dense_template['valid_pixel_ids']
    valid_pixel_3d_faces = dense_template['valid_pixel_3d_faces']
    valid_pixel_b_coords = dense_template['valid_pixel_b_coords']

    pixel_3d_points = \
        vertices[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
        vertices[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
        vertices[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]
    pixel_3d_normals = \
        normals[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
        normals[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
        normals[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]
    pixel_3d_normals = pixel_3d_normals / np.linalg.norm(pixel_3d_normals, axis=-1)[:, np.newaxis]
    displacements = displacement_map[y_coords[valid_pixel_ids].astype(int), x_coords[valid_pixel_ids].astype(int)]
    offsets = np.einsum('i,ij->ij', displacements, pixel_3d_normals)
    dense_vertices = pixel_3d_points + offsets
    return dense_vertices, dense_faces


# borrowed from https://github.com/YadiraF/PRNet/blob/master/utils/write.py
def write_obj(
    obj_name,
    vertices,
    faces,
    colors=None,
    texture=None,
    uvcoords=None,
    uvfaces=None,
    inverse_face_order=False,
    normal_map=None,
):
    ''' Save 3D face model with texture. 
    Ref: https://github.com/patrikhuber/eos/blob/bd00155ebae4b1a13b08bf5a991694d682abbada/include/eos/core/Mesh.hpp
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        faces: shape = (ntri, 3)
        texture: shape = (uv_size, uv_size, 3)
        uvcoords: shape = (nver, 2) max value<=1
    '''
    if os.path.splitext(obj_name)[-1] != '.obj':
        obj_name = obj_name + '.obj'
    mtl_name = obj_name.replace('.obj', '.mtl')
    texture_name = obj_name.replace('.obj', '.png')
    material_name = 'FaceTexture'

    faces = faces.copy()
    # mesh lab start with 1, python/c++ start from 0
    faces += 1
    if inverse_face_order:
        faces = faces[:, [2, 1, 0]]
        if uvfaces is not None:
            uvfaces = uvfaces[:, [2, 1, 0]]

    # write obj
    with open(obj_name, 'w') as f:
        # first line: write mtlib(material library)
        # f.write('# %s\n' % os.path.basename(obj_name))
        # f.write('#\n')
        # f.write('\n')
        if texture is not None:
            f.write('mtllib %s\n\n' % os.path.basename(mtl_name))

        # write vertices
        if colors is None:
            for i in range(vertices.shape[0]):
                f.write(
                    'v {} {} {}\n'.format(
                        vertices[i, 0], vertices[i, 1], vertices[i, 2],
                    )
                )
        else:
            for i in range(vertices.shape[0]):
                f.write(
                    'v {} {} {} {} {} {}\n'.format(
                        vertices[i, 0], vertices[i, 1], vertices[i, 2],
                        colors[i, 0], colors[i, 1], colors[i, 2],
                    )
                )

        # write uv coords
        if texture is None:
            for i in range(faces.shape[0]):
                f.write('f {} {} {}\n'.format(faces[i, 2], faces[i, 1], faces[i, 0]))
        else:
            for i in range(uvcoords.shape[0]):
                f.write('vt {} {}\n'.format(uvcoords[i,0], uvcoords[i,1]))
            f.write('usemtl %s\n' % material_name)
            # write f: ver ind/ uv ind
            uvfaces = uvfaces + 1
            for i in range(faces.shape[0]):
                f.write(
                    'f {}/{} {}/{} {}/{}\n'.format(
                        faces[i, 0], uvfaces[i, 0],
                        faces[i, 1], uvfaces[i, 1],
                        faces[i, 2], uvfaces[i, 2],
                    )
                )
            # write mtl
            with open(mtl_name, 'w') as f:
                f.write('newmtl %s\n' % material_name)
                s = 'map_Kd {}\n'.format(os.path.basename(texture_name)) # map to image
                f.write(s)

                if normal_map is not None:
                    name, _ = os.path.splitext(obj_name)
                    normal_name = f'{name}_normals.png'
                    f.write(f'disp {normal_name}')
                    cv2.imwrite(normal_name, normal_map)

            cv2.imwrite(texture_name, texture)


## load obj,  similar to load_obj from pytorch3d
def load_obj(obj_filename):
    """ Ref: https://github.com/facebookresearch/pytorch3d/blob/25c065e9dafa90163e7cec873dbb324a637c68b7/pytorch3d/io/obj_io.py
    Load a mesh from a file-like object.
    """
    with open(obj_filename, 'r') as f:
        lines = [line.strip() for line in f]

    verts, uvcoords = [], []
    faces, uv_faces = [], []
    # startswith expects each line to be a string. If the file is read in as
    # bytes then first decode to strings.
    if lines and isinstance(lines[0], bytes):
        lines = [el.decode("utf-8") for el in lines]

    for line in lines:
        tokens = line.strip().split()
        if line.startswith("v "):  # Line is a vertex.
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                msg = "Vertex %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)
        elif line.startswith("vt "):  # Line is a texture.
            tx = [float(x) for x in tokens[1:3]]
            if len(tx) != 2:
                raise ValueError(
                    "Texture %s does not have 2 values. Line: %s" % (str(tx), str(line))
                )
            uvcoords.append(tx)
        elif line.startswith("f "):  # Line is a face.
            # Update face properties info.
            face = tokens[1:]
            face_list = [f.split("/") for f in face]
            for vert_props in face_list:
                # Vertex index.
                faces.append(int(vert_props[0]))
                if len(vert_props) > 1:
                    if vert_props[1] != "":
                        # Texture index is present e.g. f 4/1/1.
                        uv_faces.append(int(vert_props[1]))

    verts = torch.tensor(verts, dtype=torch.float32)
    uvcoords = torch.tensor(uvcoords, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.long)
    faces = faces.reshape(-1, 3) - 1
    uv_faces = torch.tensor(uv_faces, dtype=torch.long)
    uv_faces = uv_faces.reshape(-1, 3) - 1
    return (
        verts,
        uvcoords,
        faces,
        uv_faces,
    )


# ---------------------------- process/generate vertices, normals, faces
def generate_triangles(h, w, margin_x=2, margin_y=5):
    # quad layout:
    # 0 1 ... w-1
    # w w+1
    #.
    # w*h
    triangles = []
    for x in range(margin_x, w-1-margin_x):
        for y in range(margin_y, h-1-margin_y):
            triangle0 = [y*w + x, y*w + x + 1, (y+1)*w + x]
            triangle1 = [y*w + x + 1, (y+1)*w + x + 1, (y+1)*w + x]
            triangles.append(triangle0)
            triangles.append(triangle1)
    triangles = np.array(triangles)
    triangles = triangles[:,[0,2,1]]
    return triangles


# borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
def face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]
    

def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)
    bs, nv = vertices.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None] # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.reshape(-1, 3)
    vertices_faces = vertices_faces.reshape(-1, 3, 3)

    normals.index_add_(
        0, faces[:, 1].long(), 
        torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]),
    )
    normals.index_add_(
        0, faces[:, 2].long(), 
        torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]),
    )
    normals.index_add_(
        0, faces[:, 0].long(),
        torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]),
    )

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals


def batch_orth_proj(X, camera):
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:, :, 2:]], 2)
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn


def transform_points(points, tform, points_scale=None, out_scale=None):
    points_2d = points[:,:,:2]
        
    #'input points must use original range'
    if points_scale:
        assert points_scale[0] == points_scale[1]
        points_2d = (points_2d*0.5 + 0.5) * points_scale[0]

    batch_size, n_points, _ = points.shape
    trans_points_2d = torch.bmm(
        torch.cat(
            [
                points_2d, 
                torch.ones([batch_size, n_points, 1],
                device=points.device,
                dtype=points.dtype),
            ],
            dim=-1,
        ), 
        tform,
    ) 
    # h, w of output image size
    if out_scale: 
        trans_points_2d[:,:,0] = trans_points_2d[:,:,0] / out_scale[1] * 2 - 1
        trans_points_2d[:,:,1] = trans_points_2d[:,:,1] / out_scale[0] * 2 - 1
    trans_points = torch.cat([trans_points_2d[:,:,:2], points[:,:,2:]], dim=-1)
    return trans_points


# -------------------------------------- image processing
def binary_erosion(tensor, kernel_size=5):
    # tensor: [bz, 1, h, w]. 
    device = tensor.device
    mask = tensor.cpu().numpy()
    structure=np.ones((kernel_size,kernel_size))
    new_mask = mask.copy()
    for i in range(mask.shape[0]):
        new_mask[i,0] = morphology.binary_erosion(mask[i,0], structure)
    return torch.from_numpy(new_mask.astype(np.float32)).to(device)


# -------------------------------------- io
def copy_state_dict(
    cur_state_dict,
    pre_state_dict,
    prefix='', 
    load_name=None,
):
    def _get_params(key):
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        elif "module." + key in pre_state_dict:
            # pretrained model is DataParallel, current model is not 
            return pre_state_dict["module." + key]
        elif key[7:] in pre_state_dict:
            # current model is DataParallel, pretrained model is not 
            return pre_state_dict[key[7:]]
        else:
            raise NotImplementedError(f"{key} not found!")
    
    for k in cur_state_dict:
        if load_name is not None:
            if load_name not in k:
                continue
        v = _get_params(k)
        cur_state_dict[k].copy_(v)


def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image*255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1,2,0)[:,:,[2,1,0]]
    return image.astype(np.uint8).copy()


def dict2obj(d):
    if not isinstance(d, dict):
        return d
    class C(object):
        pass

    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o


def dict_tensor2npy(tensor_dict):
    npy_dict = {}
    for key in tensor_dict:
        npy_dict[key] = tensor_dict[key][0].cpu().numpy()
    return npy_dict


# ---------------------------------- visualization
def plot_kpts(image, kpts):
    ''' Draw 68 key points
    Args: 
        image: the input image
        kpt: (68, 3).
    '''
    # color in BGR order
    color_visible = (0, 255, 0) # green
    color_invisible = (0, 0, 255) # red
    
    image = image.copy()
    kpts = kpts.copy()
    radius = max(int(min(image.shape[0], image.shape[1])/200), 1)
    for line in LINES_KPTS68:
        # plot the points first
        for i in np.unique(line):
            # if there are visibility information, we will draw landmarks in different color
            lm_color = color_invisible if (kpts.shape[1] == 4 and kpts[i, 3] <= 0.5) else color_visible
            st = kpts[i, :2].astype(int)
            image = cv2.circle(image, st, radius, lm_color, radius*2) 
        # plot the lines
        for i in range(1, len(line)):
            st = kpts[line[i - 1], :2].astype(int)
            ed = kpts[line[i], :2].astype(int)
            image = cv2.line(image, st, ed, (255, 255, 255), radius)

    return image


def plot_verts(image, kpts):
    ''' Draw 68 key points
    Args: 
        image: the input image
        kpt: (68, 3).
    '''
    # color in BGR order
    c = (255, 0, 0)
    image = image.copy()

    for i in range(kpts.shape[0]):
        st = kpts[i, :2].astype(int)
        image = cv2.circle(image, st, 1, c, 2)

    return image


def tensor_vis_landmarks(images, landmarks, gt_landmarks=None, isScale=True):
    # visualize landmarks
    vis_landmarks = []
    images = images.cpu().numpy()
    predicted_landmarks = landmarks.detach().cpu().numpy()
    if gt_landmarks is not None:
        gt_landmarks_np = gt_landmarks.detach().cpu().numpy()
    
    for i in range(images.shape[0]):
        image = images[i]
        # CHW -> HWC, then RGB -> BGR
        image = image.transpose(1, 2, 0)[..., [2, 1, 0]].copy()
        image = image * 255.
        h, w = image.shape[:2]
        if isScale:
            predicted_landmark = predicted_landmarks[i]
            predicted_landmark[..., 0] = predicted_landmark[..., 0] * w / 2 + w / 2
            predicted_landmark[..., 1] = predicted_landmark[..., 1] * h / 2 + h / 2
        else:
            predicted_landmark = predicted_landmarks[i]
        if predicted_landmark.shape[0] == 68:
            image_landmarks = plot_kpts(image, predicted_landmark)
            if gt_landmarks is not None:
                image_landmarks = plot_verts(
                    image_landmarks, gt_landmarks_np[i] * h / 2 + h / 2,
                )
        else:
            image_landmarks = plot_verts(image, predicted_landmark)
            if gt_landmarks is not None:
                image_landmarks = plot_verts(
                    image_landmarks, gt_landmarks_np[i] * h / 2 + h / 2,
                )
        vis_landmarks.append(image_landmarks)

    vis_landmarks = np.stack(vis_landmarks)
    vis_landmarks = torch.from_numpy(vis_landmarks[..., [2, 1, 0]].transpose(0, 3, 1, 2)) / 255.
    return vis_landmarks


############### for training
def load_local_mask(image_size, tdmm_type, mode='bbx'):
    if mode == 'bbx':
        # UV space face attributes bbx in size 2048 (l r t b)
        if tdmm_type == "AR":
            face = np.array([1024-750, 1024+750, 175, 175+1500])
            forehead = np.array([1024-550, 1024+550, 200, 200+400])
            eye_nose = np.array([1024-500, 1024+500, 550, 550+560])
            mouth = np.array([1024-400, 1024+400, 1050, 1050+600])
        elif tdmm_type == "FLAME":
            face = np.array([400, 1648, 400, 1648])
            forehead = np.array([550, 1498, 430, 700+50])
            eye_nose = np.array([490, 1558, 700, 1050+50])
            mouth = np.array([574, 1474, 1050, 1550])
        else:
            raise NotImplementedError(f"Unknown 3DMM type: {tdmm_type}")
        ratio = image_size / 2048.
        face = (face * ratio).astype(np.int)
        forehead = (forehead * ratio).astype(np.int)
        eye_nose = (eye_nose * ratio).astype(np.int)
        mouth = (mouth * ratio).astype(np.int)
        regional_mask = np.array([face, forehead, eye_nose, mouth])

    return regional_mask


def visualize_grid(visdict, savepath=None, size=224, dim=1, return_gird=True):
    '''
    image range should be [0,1]
    dim: 2 for horizontal. 1 for vertical
    '''
    assert dim == 1 or dim==2
    grids = {}
    for key in visdict:
        _,_,h,w = visdict[key].shape
        if dim == 2:
            new_h = size
            new_w = int(w * size / h)
        elif dim == 1:
            new_h = int(h * size / w)
            new_w = size
        grids[key] = torchvision.utils.make_grid(
            F.interpolate(visdict[key], [new_h, new_w]).detach().cpu()
        )
    grid = torch.cat(list(grids.values()), dim)
    grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[..., [2, 1, 0]]
    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
    if savepath:
        cv2.imwrite(savepath, grid_image)
    if return_gird:
        return grid_image  


# Rotation Converter
# Repre: euler angle(3), angle axis(3), rotation matrix(3x3), quaternion(4)
# ref: https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/core/conversions.html#
def deg2rad(tensor):
    """Function that converts angles from degrees to radians.
    See :class:`~torchgeometry.DegToRad` for details.
    Args:
        tensor (Tensor): Tensor of arbitrary shape.
    Returns:
        Tensor: Tensor with same shape as input.
    Examples::
        >>> input = 360. * torch.rand(1, 3, 3)
        >>> output = tgm.deg2rad(input)
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))

    return tensor * PI.to(tensor.device).type(tensor.dtype) / 180.


def euler_to_quaternion(r):
    x = r[..., 0]
    y = r[..., 1]
    z = r[..., 2]

    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = torch.cos(z)
    sz = torch.sin(z)
    cy = torch.cos(y)
    sy = torch.sin(y)
    cx = torch.cos(x)
    sx = torch.sin(x)
    quaternion = torch.zeros_like(r.repeat(1,2))[..., :4].to(r.device)
    quaternion[..., 0] += cx*cy*cz - sx*sy*sz
    quaternion[..., 1] += cx*sy*sz + cy*cz*sx
    quaternion[..., 2] += cx*cz*sy - sx*cy*sz
    quaternion[..., 3] += cx*cy*sz + sx*cz*sy
    return quaternion


def quaternion_to_angle_axis(quaternion: torch.Tensor):
    """Convert quaternion vector to angle axis of rotation. 
    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h
    Args:
        quaternion (torch.Tensor): tensor with quaternions.
    Return:
        torch.Tensor: tensor with angle axis of rotation.
    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`
    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta).to(quaternion.device)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion).to(quaternion.device)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


#### batch converter
def batch_euler2axis(r):
    return quaternion_to_angle_axis(euler_to_quaternion(r))