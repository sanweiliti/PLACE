import torch.nn.functional as F
import torch
import sys, os, glob
import json
import numpy as np
import torch.nn as nn
import torchgeometry as tgm
import logging
import datetime
import random
from scipy.spatial.transform import Rotation as R
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    # def forward(self, module_input):
    #     reshaped_input = module_input.view(-1, 3, 2)
    #     b1 = F.normalize(reshaped_input[:, :, 0], dim=1)
    #     dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
    #     b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
    #     b3 = torch.cross(b1, b2, dim=1)
    #
    #     return torch.stack([b1, b2, b3], dim=-1)


    @staticmethod
    def decode(module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


    @staticmethod
    def matrot2aa(pose_matrot):  # input: [bs, 3, 3]
        '''
        :param pose_matrot: Nx1xnum_jointsx9
        :return: Nx1xnum_jointsx3
        '''

        homogen_matrot = F.pad(pose_matrot.view(-1, 3, 3), [0,1])  # [bs, 3, 4], float
        pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(-1, 3).contiguous()
        return pose

    @staticmethod
    def aa2matrot(pose):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nx1xnum_jointsx9
        '''
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous()
        return pose_body_matrot



def verts_transform(verts_batch, cam_ext_batch):
    # transfrom verts to world coornidate
    verts_batch_homo = F.pad(verts_batch, (0, 1), mode='constant', value=1)
    verts_batch_homo_transformed = torch.matmul(verts_batch_homo,
                                                cam_ext_batch.permute(0, 2, 1))
    verts_batch_transformed = verts_batch_homo_transformed[:, :, :-1]
    return verts_batch_transformed


def get_contact_id(body_segments_folder, contact_body_parts=['L_Hand', 'R_Hand']):
    contact_verts_ids = []
    contact_faces_ids = []

    for part in contact_body_parts:
        with open(os.path.join(body_segments_folder, part + '.json'), 'r') as f:
            data = json.load(f)
            contact_verts_ids.append(list(set(data["verts_ind"])))
            contact_faces_ids.append(list(set(data["faces_ind"])))

    contact_verts_ids = np.concatenate(contact_verts_ids)
    contact_faces_ids = np.concatenate(contact_faces_ids)
    return contact_verts_ids, contact_faces_ids


def normalize_global_T(x_batch, cam_intrisic, max_depth):
    '''
    according to the camera intrisics and maximal depth,
    normalize the global translate to [-1, 1] for X, Y and Z.
    input: [transl, rotation, local params]
    '''
    xt_batch = x_batch[:, :3]
    xr_batch = x_batch[:, 3:]

    fx_batch = cam_intrisic[:, 0, 0]
    fy_batch = cam_intrisic[:, 1, 1]
    px_batch = cam_intrisic[:, 0, 2]
    py_batch = cam_intrisic[:, 1, 2]
    s_ = 1.0 / torch.max(px_batch, py_batch)
    x = s_ * xt_batch[:, 0] * fx_batch / (xt_batch[:, 2] + 1e-6)
    y = s_ * xt_batch[:, 1] * fy_batch / (xt_batch[:, 2] + 1e-6)

    z = 2.0 * xt_batch[:, 2] / max_depth - 1.0
    xt_batch_normalized = torch.stack([x, y, z], dim=-1)
    return torch.cat([xt_batch_normalized, xr_batch], dim=-1)


def recover_global_T(x_batch, cam_intrisic, max_depth):
    '''
    according to the camera intrisics and maximal depth,
    recover the translate from [-1, 1] for X, Y and Z, to the physical unit.
    input: [transl, rotation, local params]
    '''
    xt_batch = x_batch[:, :3]
    xr_batch = x_batch[:, 3:]

    fx_batch = cam_intrisic[:, 0, 0]
    fy_batch = cam_intrisic[:, 1, 1]
    px_batch = cam_intrisic[:, 0, 2]
    py_batch = cam_intrisic[:, 1, 2]
    s_ = 1.0 / torch.max(px_batch, py_batch)

    z = (xt_batch[:, 2] + 1.0) / 2.0 * max_depth

    x = xt_batch[:, 0] * z / s_ / fx_batch
    y = xt_batch[:, 1] * z / s_ / fy_batch

    xt_batch_recoverd = torch.stack([x, y, z], dim=-1)
    return torch.cat([xt_batch_recoverd, xr_batch], dim=-1)


def convert_to_6D_rot(x_batch):
    '''
    input: [transl, rotation, local params]
    convert global rotation from Eular angle to 6D continuous representation
    '''

    xt = x_batch[:, :3]
    xr = x_batch[:, 3:6]
    xb = x_batch[:, 6:]

    xr_mat = ContinousRotReprDecoder.aa2matrot(xr)  # return [:,3,3]
    xr_repr = xr_mat[:, :, :-1].reshape([-1, 6])
    return torch.cat([xt, xr_repr, xb], dim=-1)


def convert_to_3D_rot(x_batch):
    '''
    input: [transl, 6d rotation, local params]
    convert global rotation from 6D continuous representation to Eular angle
    '''
    xt = x_batch[:, :3]  # (reconstructed) normalized global translation
    xr = x_batch[:, 3:9]  # (reconstructed) 6D rotation vector
    xb = x_batch[:, 9:]  # pose $ shape parameters

    xr_mat = ContinousRotReprDecoder.decode(xr)  # [bs,3,3]
    xr_aa = ContinousRotReprDecoder.matrot2aa(xr_mat)  # return [:,3]
    return torch.cat([xt, xr_aa, xb], dim=-1)


def get_logger(logdir):
    logger = logging.getLogger('emotion')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

def save_config(logdir, config):
    param_path = os.path.join(logdir, "params.json")
    print("[*] PARAM path: %s" % param_path)
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    image_paths = []
    for looproot, _, filenames in os.walk(rootdir):
        for filename in filenames:
            if filename.endswith(suffix):
                image_paths.append(os.path.join(looproot, filename))
    return image_paths


def body_params_encapsulate_batch(x_body_rec):
    body_params_batch_rec = {}
    body_params_batch_rec['transl'] = x_body_rec[:, :3]
    body_params_batch_rec['global_orient'] = x_body_rec[:, 3:6]
    body_params_batch_rec['betas'] = x_body_rec[:, 6:16]
    body_params_batch_rec['body_pose_vp'] = x_body_rec[:, 16:48]
    body_params_batch_rec['left_hand_pose'] = x_body_rec[:, 48:60]
    body_params_batch_rec['right_hand_pose'] = x_body_rec[:, 60:]
    return body_params_batch_rec


def body_params_parse(body_params_batch):
    '''
    input:  body_params
                    |-- transl: global translation, [1, 3D]
                    |-- global_orient: global rotation, [1, 3D]
                    |-- betas:  body shape, [1, 10D]
                    |-- body_pose:  in Vposer latent space, [1, 32D]
                    |-- left_hand_pose: [1, 12]
                    |-- right_hand_pose: [1, 12]
                    |-- camera_translation: [1, 3D]
                    |-- camera_rotation: [1, 3x3 mat]
            z_s: scene representation [1, 128D]
    '''
    ## parse body_params_batch
    x_body_T = body_params_batch['transl']
    x_body_R = body_params_batch['global_orient']
    x_body_beta = body_params_batch['betas']
    x_body_pose = body_params_batch['body_pose_vp']
    x_body_lh = body_params_batch['left_hand_pose']
    x_body_rh = body_params_batch['right_hand_pose']

    x_body = np.concatenate([x_body_T, x_body_R,
                             x_body_beta, x_body_pose,
                             x_body_lh, x_body_rh], axis=-1)
    return x_body


def gen_body_mesh(body_params, body_pose_joint, smplx_model):
    # input: body_params/body_pose_joint: tensor batch
    # output: body_verts: np array batch
    body_params_dict = body_params_encapsulate_batch(body_params)
    body_param_ = {}
    for key in body_params_dict.keys():
        if key in ['body_pose_vp']:
            continue
        else:
            body_param_[key] = body_params_dict[key]
    smplx_output = smplx_model(return_verts=True, body_pose=body_pose_joint, **body_param_)  # generated human body mesh
    body_verts = smplx_output.vertices  # [bs, n_body_vert, 3]
    return body_verts


def update_globalRT_for_smplx(body_params, smplx_model, trans_to_target_origin, delta_T=None):
    '''
    input:
        body_params: array, [72], under camera coordinate
        smplx_model: the model to generate smplx mesh, given body_params
        trans_to_target_origin: coordinate transformation [4,4] mat
    Output:
        body_params with new globalR and globalT, which are corresponding to the new coord system
    '''

    ### step (1) compute the shift of pelvis from the origin
    body_params_dict = {}
    body_params_dict['transl'] = np.expand_dims(body_params[:3], axis=0)
    body_params_dict['global_orient'] = np.expand_dims(body_params[3:6], axis=0)
    body_params_dict['betas'] = np.expand_dims(body_params[6:16], axis=0)
    body_params_dict['body_pose_vp'] = np.expand_dims(body_params[16:48], axis=0)
    body_params_dict['left_hand_pose'] = np.expand_dims(body_params[48:60], axis=0)
    body_params_dict['right_hand_pose'] = np.expand_dims(body_params[60:], axis=0)

    body_param_dict_torch = {}
    for key in body_params_dict.keys():
        body_param_dict_torch[key] = torch.FloatTensor(body_params_dict[key]).to(device)

    if delta_T is None:
        body_param_dict_torch['transl'] = torch.zeros([1,3], dtype=torch.float32).to(device)
        body_param_dict_torch['global_orient'] = torch.zeros([1,3], dtype=torch.float32).to(device)
        smplx_out = smplx_model(return_verts=True, **body_param_dict_torch)
        delta_T = smplx_out.joints[0,0,:] # (3,)
        delta_T = delta_T.detach().cpu().numpy()

    ### step (2): calibrate the original R and T in body_params
    body_R_angle = body_params_dict['global_orient'][0]
    body_R_mat = R.from_rotvec(body_R_angle).as_dcm() # to a [3,3] rotation mat
    body_T = body_params_dict['transl'][0]
    body_mat = np.eye(4)
    body_mat[:-1,:-1] = body_R_mat
    body_mat[:-1, -1] = body_T + delta_T

    ### step (3): perform transformation, and decalib the delta shift
    body_params_dict_new = copy.deepcopy(body_params_dict)
    body_mat_new = np.dot(trans_to_target_origin, body_mat)
    body_R_new = R.from_dcm(body_mat_new[:-1,:-1]).as_rotvec()
    body_T_new = body_mat_new[:-1, -1]
    body_params_dict_new['global_orient'] = body_R_new.reshape(1,3)
    body_params_dict_new['transl'] = (body_T_new - delta_T).reshape(1,3)
    body_param_new = body_params_parse(body_params_dict_new)[0]  # array, [72]
    return body_param_new
