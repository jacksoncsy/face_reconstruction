import torch
import torch.nn as nn
import numpy as np
import pickle
import os.path as osp

from .tdmm_utils import (
    batch_rodrigues,
    batch_rotvec2matrix,
    lbs,
    load_obj,
    matrix2angle,
    rot_mat_to_euler,
    to_np,
    to_tensor,
    vertices2landmarks,
    Struct,    
)


class ARMultilinear(nn.Module):
    def __init__(self, tdmm_dir):
        super(ARMultilinear, self).__init__()

        self.dtype = torch.float32
        # load multilinear model bases
        u_core, u_id, u_exp = self.load_basis(osp.join(tdmm_dir, "AR_multilinear.bin"))
        self.register_buffer('u_core', torch.tensor(u_core, dtype=self.dtype))
        self.register_buffer('u_id', torch.tensor(u_id, dtype=self.dtype))
        self.register_buffer('u_exp', torch.tensor(u_exp, dtype=self.dtype))

        # compute average ID parameters
        avg_w_id = np.mean(u_id, axis=0)
        self.register_buffer('avg_w_id', torch.tensor(avg_w_id, dtype=self.dtype))
        # compute covariance matrix for identity weights
        inv_cov_id = np.linalg.inv(np.cov(u_id.T))
        self.register_buffer('inv_cov_id', torch.tensor(inv_cov_id, dtype=self.dtype))
        
        # load template
        verts, _, faces, _ = load_obj(osp.join(tdmm_dir, "base.obj"))
        # vertices
        self.register_buffer('v_template', verts)
        # triangles
        self.register_buffer('faces_tensor', faces)
        # Triangulation
        self.trilist = to_np(faces, dtype=np.int64)

        # load static and dynamic landmark embeddings
        self.load_landmark_embeddings(osp.join(tdmm_dir, "landmark_embedding.pkl"))

    def load_landmark_embeddings(self, filepath):
        lmk_embeddings = pickle.load(open(filepath, "rb"))
        # (51,)
        self.register_buffer(
            'lmk_faces_idx', 
            torch.from_numpy(lmk_embeddings['static_lmk_faces_idx']).long(),
        )
        # (51, 3)
        self.register_buffer(
            'lmk_bary_coords',
            torch.from_numpy(lmk_embeddings['static_lmk_bary_coords']).to(self.dtype),
        )
        # (181, 17)
        self.register_buffer(
            'dynamic_lmk_faces_idx',
            torch.from_numpy(lmk_embeddings['dynamic_lmk_faces_idx']).long(),
        )
        # (181, 17, 3)
        self.register_buffer(
            'dynamic_lmk_bary_coords',
            torch.from_numpy(lmk_embeddings['dynamic_lmk_bary_coords']).to(self.dtype),
        )
        # (1, 68)
        self.register_buffer(
            'full_lmk_faces_idx',
            torch.from_numpy(lmk_embeddings['full_lmk_faces_idx']).long(),
        )
        # (1, 68, 3)
        self.register_buffer(
            'full_lmk_bary_coords',
            torch.from_numpy(lmk_embeddings['full_lmk_bary_coords']).to(self.dtype),
        )
        
    def load_basis(self, model_path):
        with open(model_path, "rb") as f:
            num_ids, num_exps, num_verts = np.fromfile(f, np.int32, 3)
            u_core = np.fromfile(f, np.float32, num_ids * num_exps * num_verts)
            u_core = u_core.reshape(num_ids, num_exps, num_verts)
            
            m, n = np.fromfile(f, np.int32, 2)
            u_id = np.fromfile(f, np.float32, m * n)
            u_id = u_id.reshape(m, n)

            m, n = np.fromfile(f, np.int32, 2)
            u_exp = np.fromfile(f, np.float32, m * n)
            u_exp = u_exp.reshape(m, n)

            # expand the order-3 tensor into a order-2 tensor for easier manipulation
            u_core = u_core.reshape(u_id.shape[1], -1)
            
            # Make the tensors/matrices computation friendly
            # https://pybind11.readthedocs.io/en/master/advanced/cast/eigen.html#pass-by-reference
            u_core = np.asfortranarray(u_core)
            u_id = np.asfortranarray(u_id)
            u_exp = np.asfortranarray(u_exp)

        return u_core, u_id, u_exp

    def _find_dynamic_lmk_idx_and_bcoords(
        self, pose, dynamic_lmk_faces_idx, dynamic_lmk_b_coords, dtype=torch.float32,
    ):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: (N, 3)
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
        """
        # get yaw angle
        # (N, 3, 3)
        rot_mats = batch_rodrigues(pose, dtype=dtype)
        # (N,), restrict the angle within [-90, 90]
        yaw_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rot_mats)*180.0/np.pi, min=-90, max=90)
        ).to(dtype=torch.long)
        # select from the list
        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx, 0, yaw_angle + 90)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, yaw_angle + 90)

        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def seletec_3d68(self, vertices):
        landmarks3d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
            self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1),
        )
        return landmarks3d

    def get_euler_angles(self, pose_params, dtype=torch.float32):
        """
            Input:
                pose_params: (bs, 6)
            return:
                angles: (bs, 3), [yaw, pitch, roll]
        """
        batch_size = pose_params.shape[0]
        # get the final rotation matrix
        # (bs, 3, 3)
        rot_mats = batch_rodrigues(pose_params[:, :3])
        # get the angles
        angles = torch.zeros((batch_size, 3), dtype=dtype)
        for idx in range(batch_size):
            yaw, pitch, roll = matrix2angle(rot_mats[idx])
            angles[idx, 0] = yaw
            angles[idx, 1] = pitch
            angles[idx, 2] = roll
        
        return angles

    def forward(self, shape_params, expression_params, pose_params):
        """
            Input:
                shape_params: (N, n_shape)
                expression_params: (N, n_exp)
                pose_params: (N, n_pose), n_pose=6, rotation vector (axis-angle) + [tx, ty, tx]
            return:
                vertices: (bs, nv, 3)
                landmarks2d: (bs, n_lmk, 3), 2D-style landmarks
                landmarks3d: (bs, n_lmk, 3), full 3D landmarks
                face_poses: (bs, 3), face pose in radians with yaw-pitch-roll order            
        """
        batch_size = shape_params.shape[0]
        # project shape parameters first
        m0 = torch.matmul(shape_params, self.u_core)
        # (N, 50, V*3)
        m0 = m0.reshape(batch_size, self.u_exp.shape[1], -1)
        # (N, 53)
        exp_tensor = torch.cat(
            [
                1 - torch.sum(expression_params, axis=1, keepdim=True),
                expression_params,
            ],
            axis=1,
        )
        # (N, 1, 50), project expression parameters
        w_exp = torch.matmul(exp_tensor, self.u_exp)[:, None]
        m1 = torch.bmm(w_exp, m0)
        # get face vertices
        vertices = m1.reshape(batch_size, -1, 3)

        # rotate the mesh globally using extrinsic camera matrix
        R = batch_rotvec2matrix(pose_params[:, :3])
        vertices = torch.bmm(vertices, R.transpose(1, 2)) + pose_params[:, None, 3:]
        
        # (N, 51)
        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        # (N, 51, 3)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)
        # get indices for the boundary points
        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            pose_params[:, :3],
            self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            dtype=self.dtype,
        )
        # (N, 68)
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        # (N, 68, 3)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)
        
        # get 2d landmarks (face boundary is on the visible contour)
        landmarks2d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            lmk_faces_idx,
            lmk_bary_coords,
        )
        
        # get 3d landmarks
        landmarks3d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.repeat(batch_size, 1),
            self.full_lmk_bary_coords.repeat(batch_size, 1, 1),
        )

        # get face pose
        face_poses = self.get_euler_angles(pose_params)

        return vertices, landmarks2d, landmarks3d, face_poses


class FLAME(nn.Module):
    """
    borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """
    def __init__(self, tdmm_dir, n_shape, n_exp):
        super(FLAME, self).__init__()
        print("creating the FLAME Decoder")
        with open(osp.join(tdmm_dir, "generic_model.pkl"), 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)

        self.dtype = torch.float32
        self.register_buffer(
            'faces_tensor', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long),
        )
        # Triangulation
        self.trilist = to_np(flame_model.f, dtype=np.int64)
        # The vertices of the template model
        self.register_buffer(
            'v_template', to_tensor(to_np(flame_model.v_template), dtype=self.dtype),
        )
        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:,:,:n_shape], shapedirs[:,:,300:300+n_exp]], 2)
        self.register_buffer('shapedirs', shapedirs)
        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype))
        # 
        self.register_buffer(
            'J_regressor', to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype),
        )
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer(
            'lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype),
        )

        # Fixing Eyeball and neck rotation
        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter(
            'eye_pose', nn.Parameter(default_eyball_pose, requires_grad=False),
        )

        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter(
            'neck_pose', nn.Parameter(default_neck_pose, requires_grad=False),
        )

        # Static and Dynamic Landmark embeddings for FLAME
        lmk_embeddings = np.load(
            osp.join(tdmm_dir, "landmark_embedding.npy"),
            allow_pickle=True,
            encoding='latin1',
        )
        lmk_embeddings = lmk_embeddings[()]
        # (51,)
        self.register_buffer(
            'lmk_faces_idx', torch.from_numpy(lmk_embeddings['static_lmk_faces_idx']).long(),
        )
        # (51, 3)
        self.register_buffer(
            'lmk_bary_coords', torch.from_numpy(lmk_embeddings['static_lmk_bary_coords']).to(self.dtype),
        )
        # (79, 17)
        self.register_buffer(
            'dynamic_lmk_faces_idx', lmk_embeddings['dynamic_lmk_faces_idx'].long(),
        )
        # (79, 17, 3)
        self.register_buffer(
            'dynamic_lmk_bary_coords', lmk_embeddings['dynamic_lmk_bary_coords'].to(self.dtype),
        )
        # (1, 68)
        self.register_buffer(
            'full_lmk_faces_idx', torch.from_numpy(lmk_embeddings['full_lmk_faces_idx']).long(),
        )
        # (1, 68, 3)
        self.register_buffer(
            'full_lmk_bary_coords', torch.from_numpy(lmk_embeddings['full_lmk_bary_coords']).to(self.dtype),
        )

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))
        
    def _find_dynamic_lmk_idx_and_bcoords(
        self, pose, dynamic_lmk_faces_idx, dynamic_lmk_b_coords, neck_kin_chain, dtype=torch.float32,
    ):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
        """

        batch_size = pose.shape[0]

        aa_pose = torch.index_select(
            pose.view(batch_size, -1, 3), 1, neck_kin_chain,
        )

        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3), dtype=dtype,
        ).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(
            3, device=pose.device, dtype=dtype,
        ).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        
        # get the final rotation matrix
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        # get yaw angle
        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)
        ).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        # mapping from old to new is:
        # old: [..., -40, -39, -38, -37, ..., -2, -1, 0, 1, 2, ..., 38, 39]
        # new: [...,  78,  78,  77,  76, ..., 41, 40, 0, 1, 2, ..., 38, 39]
        y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx, 0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)

        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def seletec_3d68(self, vertices):
        landmarks3d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
            self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1),
        )
        return landmarks3d

    def get_euler_angles(self, pose_params, dtype=torch.float32):
        """
            Input:
                pose_params:  (bs, 6)
            return:
                angles: (bs, 3), [yaw, pitch, roll]
        """
        batch_size = pose_params.shape[0]
        # no eye pose in current method, so we use default ones
        eye_pose_params = self.eye_pose.expand(batch_size, -1)
        full_pose = torch.cat(
            [
                pose_params[:, :3],
                self.neck_pose.expand(batch_size, -1),
                pose_params[:, 3:],
                eye_pose_params,
            ],
            dim=1,
        )

        aa_pose = torch.index_select(
            full_pose.view(batch_size, -1, 3), 1, self.neck_kin_chain,
        )

        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3), dtype=dtype,
        ).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(
            3, device=full_pose.device, dtype=dtype,
        ).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        
        # get the final rotation matrix
        # (bs, 3, 3)
        for idx in range(len(self.neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        # get the angles
        angles = torch.zeros((batch_size, 3), dtype=dtype)
        for idx in range(batch_size):
            angles[idx] = matrix2angle(rel_rot_mat[idx])

        return angles

    def forward(self, shape_params, expression_params, pose_params):
        """
            Input:
                shape_params: (bs, number of shape parameters)
                expression_params: (bs, number of expression parameters)
                pose_params: (bs, number of pose parameters==6)
            return:
                vertices: (bs, nv, 3)
                landmarks2d: (bs, n_lmk, 3), 2D-style landmarks
                landmarks3d: (bs, n_lmk, 3), full 3D landmarks
                face_poses: (bs, 3), face pose in radians with yaw-pitch-roll order
        """
        batch_size = shape_params.shape[0]
        # no eye pose in current method, so we use default ones
        eye_pose_params = self.eye_pose.expand(batch_size, -1)
        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat(
            [
                pose_params[:, :3],
                self.neck_pose.expand(batch_size, -1),
                pose_params[:, 3:],
                eye_pose_params,
            ],
            dim=1,
        )
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        vertices, _ = lbs(
            betas, full_pose, template_vertices,
            self.shapedirs, self.posedirs,
            self.J_regressor, self.parents,
            self.lbs_weights, dtype=self.dtype,
        )

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)
        
        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            full_pose, 
            self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain,
            dtype=self.dtype,
        )
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks2d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            lmk_faces_idx,
            lmk_bary_coords,
        )
        
        landmarks3d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.repeat(batch_size, 1),
            self.full_lmk_bary_coords.repeat(batch_size, 1, 1),
        )

        # get face pose
        face_poses = self.get_euler_angles(pose_params)

        return vertices, landmarks2d, landmarks3d, face_poses