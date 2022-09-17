'''
Default config for DECA
'''
from yacs.config import CfgNode as CN
import argparse
import os
import os.path as osp


cfg = CN()

cfg.deca_dir = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
cfg.device = 'cuda'
cfg.device_ids = [] # or [0, 1, 2, 3, 4, 5, 6, 7]
cfg.num_nodes = 1 # number of computing nodes to use (each node has 8 GPUs)

cfg.pretrained_modelpath = ""
cfg.output_dir = ''
cfg.rasterizer_type = 'pytorch3d'
# directory of now evaluation codes
cfg.now_codedir = "/fsx/shiyangc/projects/now_evaluation"

# ---------------------------------------------------------------------------- #
# Options for Face model
# ---------------------------------------------------------------------------- #
cfg.model = CN()
cfg.model.topology_path = osp.join(cfg.deca_dir, "models/flame/head_template.obj")
# texture data original from http://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_texture_data.zip
cfg.model.dense_template_path = osp.join(cfg.deca_dir, "models/flame/texture_data_256.npy")
cfg.model.fixed_displacement_path = osp.join(cfg.deca_dir, "models/flame/fixed_displacement_256.npy")
cfg.model.tdmm_model_path = osp.join(cfg.deca_dir, "models/flame/generic_model.pkl")
cfg.model.lmk_embedding_path = osp.join(cfg.deca_dir, "models/flame/landmark_embedding.npy")
cfg.model.face_mask_path = osp.join(cfg.deca_dir, "models/flame/uv_face_mask.png")
cfg.model.face_eye_mask_path = osp.join(cfg.deca_dir, "models/flame/uv_face_eye_mask.png")
cfg.model.mean_tex_path = osp.join(cfg.deca_dir, "models/flame/mean_texture.jpg")
cfg.model.tex_path = osp.join(cfg.deca_dir, "models/flame/FLAME_albedo_from_BFM.npz")
cfg.model.tdmm_type = 'FLAME' # FLAME, AR
cfg.model.uv_size = 256
cfg.model.param_list = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
cfg.model.n_shape = 100
cfg.model.n_tex = 50
cfg.model.n_exp = 50
cfg.model.n_cam = 3
cfg.model.n_pose = 6
cfg.model.n_light = 27
cfg.model.backbone = "resnet50" # resnet50 or mobilenetv2
cfg.model.use_tex = True
# TODO: investigate why euler did not work at all...
cfg.model.jaw_type = 'aa' # default use axis angle, another option: euler. Note that: aa is not stable in the beginning
# face recognition model
cfg.model.fr_model_path = osp.join(cfg.deca_dir, "models/pretrained/resnet50_ft_weight.pkl")
## details
cfg.model.n_detail = 128
cfg.model.max_z = 0.01

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.training_data = ['vggface2', 'ethnicity']
cfg.dataset.eval_data = ['aflw2000']
cfg.dataset.batch_size = -1
cfg.dataset.K = -1
cfg.dataset.num_workers = 16
cfg.dataset.image_size = 224
cfg.dataset.scale_min = 1.4
cfg.dataset.scale_max = 1.8
cfg.dataset.trans_scale = 0.
# path for vggface2 data
cfg.dataset.vggface2_train_image_dir = "/fsx/shiyangc/data/vggface2_processed/train/images_sampled_full"
cfg.dataset.vggface2_train_label_dir = "/fsx/shiyangc/data/vggface2_processed/train/images_sampled_full_bbox_lms_seg"
cfg.dataset.vggface2_train_cleanlist = "/fsx/shiyangc/data/vggface2_processed/train/cleanlist_v2_images_sampled_full.pkl.gz"
cfg.dataset.vggface2_eval_image_dir = "/fsx/shiyangc/data/vggface2_processed/test/images_sampled_full"
cfg.dataset.vggface2_eval_label_dir = "/fsx/shiyangc/data/vggface2_processed/test/images_sampled_full_bbox_lms_seg"
cfg.dataset.vggface2_eval_cleanlist = "/fsx/shiyangc/data/vggface2_processed/test/cleanlist_v2_images_sampled_full.pkl.gz"
# path for voxceleb2 data
cfg.dataset.voxceleb2_image_dir = "/fsx/shiyangc/data/voxceleb2_processed/dev/images_sampled"
cfg.dataset.voxceleb2_label_dir = "/fsx/shiyangc/data/voxceleb2_processed/dev/images_sampled_bbox_lms_seg"
cfg.dataset.voxceleb2_cleanlist = "/fsx/shiyangc/data/voxceleb2_processed/dev/cleanlist_v2_images_sampled.pkl.gz"
# path for BUPT-balanced-face data
cfg.dataset.ethnicity_image_dir = "/fsx/shiyangc/data/bupt_bf_processed/images_asian_african_full"
cfg.dataset.ethnicity_label_dir = "/fsx/shiyangc/data/bupt_bf_processed/images_asian_african_full_bbox_lms_seg"
cfg.dataset.ethnicity_cleanlist = "/fsx/shiyangc/data/bupt_bf_processed/cleanlist_v2_images_asian_african_full.pkl.gz"
# path for aflw2000 data
cfg.dataset.aflw_data_dir = "/fsx/shiyangc/data/aflw2000_3d/gt"
# path for NoW data
cfg.dataset.now_data_dir = "/fsx/shiyangc/data/now"

# ---------------------------------------------------------------------------- #
# Options for training
# ---------------------------------------------------------------------------- #
cfg.train = CN()
cfg.train.train_detail = False
cfg.train.max_epochs = 10
cfg.train.lr = 1e-4
cfg.train.use_lr_scheduler = True
cfg.train.log_every_n_steps = 100
cfg.train.val_every_n_steps = 500
cfg.train.vis_every_n_steps = 500
cfg.train.eval_on_now = True
cfg.train.resume = True
cfg.train.write_summary = True

# ---------------------------------------------------------------------------- #
# Options for Losses
# ---------------------------------------------------------------------------- #
cfg.loss = CN()
cfg.loss.lmk = 2.0      # face landmark loss
cfg.loss.useWlmk = True # use weighted landmarks
cfg.loss.eyed = 0.5     # eye closure loss
cfg.loss.lipd = 0.5     # lip closure loss
cfg.loss.photo = 2.0    # photemetric loss
cfg.loss.useSeg = True  # use segmentation mask
cfg.loss.id = 0.2       # identity loss using vggface feature
cfg.loss.reg_shape = 0.0005  # regularisation on shape params
cfg.loss.reg_exp = 0.001     # regularisation on expression params
cfg.loss.reg_tex = 0.001     # regularisation on texture params
cfg.loss.reg_light = 1.      # regularisation on light params
cfg.loss.shape_consistency = False  # shape consistency
cfg.loss.reg_jawpose_roll = 100. # TODO: investigate the weight for this regularisation
cfg.loss.reg_jawpose_close = 10. # TODO: investigate the weight for this regularisation

# loss for detail
cfg.loss.detail_consistency = False  # detail code consistency
cfg.loss.mrf = 3.       # ID-MRF loss
cfg.loss.photo_D = 2.   # photometric detail loss
cfg.loss.reg_sym = 0.1  # soft symmetry loss
cfg.loss.reg_z = 0.005  # regularisation on displacement values
cfg.loss.reg_diff = 0.5 # loss weights on smoothness of shading
cfg.loss.remove_outlier = False # whether to remove failure examples when computing loss


def check_cfg(cfg):
    assert cfg.train.log_every_n_steps > 0
    assert cfg.train.val_every_n_steps > 0    
    assert cfg.train.vis_every_n_steps > 0


def get_cfg_defaults():
    """ Get a yacs CfgNode object with default values for my_project.
        Return a clone so that the defaults will not be altered
        This is for the "local variable" use pattern
    """
    return cfg.clone()


def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--mode', type=str, default='train', help='deca mode')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg = get_cfg_defaults()
    cfg.cfg_file = None
    cfg.mode = args.mode
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    # sanity check of config
    check_cfg(cfg)

    return cfg
