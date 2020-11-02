import warnings
warnings.simplefilter("ignore", UserWarning)

import argparse
import torch
from tqdm import tqdm
import smplx
import math
import scipy.cluster
from scipy.stats import entropy
from utils import *
from utils_read_data import *




parser = argparse.ArgumentParser()

parser.add_argument('--smplx_model_path', type=str,
                    default='/mnt/hdd/PROX/body_models/smplx_model',
                    help='path to smplx body model')
parser.add_argument('--dataset', type=str, default='prox', help='choose dataset (prox/mp3d/replica)')
parser.add_argument('--optimize_result_dir', type=str, default='optimize_results/prox')


args = parser.parse_args()


def optimize_visulize(scene_name):
    smplx_model = smplx.create(args.smplx_model_path, model_type='smplx',
                               gender='neutral', ext='npz',
                               num_pca_comps=12,
                               create_global_orient=True,
                               create_body_pose=True,
                               create_betas=True,
                               create_left_hand_pose=True,
                               create_right_hand_pose=True,
                               create_expression=True,
                               create_jaw_pose=True,
                               create_leye_pose=True,
                               create_reye_pose=True,
                               create_transl=True,
                               batch_size=1
                               ).to(device)
    print('[INFO] smplx model loaded.')


    ##################### load optimized data ##################
    shift_list = np.load('{}/{}/shift_list.npy'.format(args.optimize_result_dir, scene_name))
    rot_angle_list_1 = np.load('{}/{}/rot_angle_list_1.npy'.format(args.optimize_result_dir, scene_name))

    body_params_opt_list_s1 = np.load('{}/{}/body_params_opt_list_s1.npy'.format(args.optimize_result_dir, scene_name))
    body_params_opt_list_s2 = np.load('{}/{}/body_params_opt_list_s2.npy'.format(args.optimize_result_dir, scene_name))
    body_verts_sample_list = np.load('{}/{}/body_verts_sample_list.npy'.format(args.optimize_result_dir, scene_name))
    n_sample = len(body_verts_sample_list)


    ####################### transfrom body params to prox coodinate system ####################
    body_params_prox_list_s1, body_params_prox_list_s2 = [], []
    for cnt in tqdm(range(0, n_sample)):
        body_params_opt_s1 = torch.from_numpy(body_params_opt_list_s1[cnt]).float().unsqueeze(0).to(device)  # tensor, [1,75]
        body_params_opt_s1 = convert_to_3D_rot(body_params_opt_s1)  # tensor, [bs=1, 72]
        body_params_opt_s2 = torch.from_numpy(body_params_opt_list_s2[cnt]).float().unsqueeze(0).to(device)  # tensor, [1,75]
        body_params_opt_s2 = convert_to_3D_rot(body_params_opt_s2)  # tensor, [bs=1, 72]
        trans_matrix_1 = np.array([[math.cos(-rot_angle_list_1[cnt]), -math.sin(-rot_angle_list_1[cnt]), 0, 0],
                                   [math.sin(-rot_angle_list_1[cnt]), math.cos(-rot_angle_list_1[cnt]), 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])
        trans_matrix_2 = np.array([[1, 0, 0, -shift_list[cnt][0]],
                                   [0, 1, 0, -shift_list[cnt][1]],
                                   [0, 0, 1, -shift_list[cnt][2]],
                                   [0, 0, 0, 1]])
        # stage 1: simple optimization
        body_params_prox_s1 = update_globalRT_for_smplx(body_params_opt_s1[0].cpu().numpy(), smplx_model, trans_matrix_2)  # [72]
        body_params_prox_s1 = update_globalRT_for_smplx(body_params_prox_s1, smplx_model, trans_matrix_1)  # [72]
        body_params_prox_list_s1.append(body_params_prox_s1)
        # stage 2: advanced optimization
        body_params_prox_s2 = update_globalRT_for_smplx(body_params_opt_s2[0].cpu().numpy(), smplx_model, trans_matrix_2)  # [72]
        body_params_prox_s2 = update_globalRT_for_smplx(body_params_prox_s2, smplx_model, trans_matrix_1)  # [72]
        body_params_prox_list_s2.append(body_params_prox_s2)
    return body_params_prox_list_s2


def diversity(body_params_prox_list, cls_num=20):
    ## k-means
    codes, dist = scipy.cluster.vq.kmeans(body_params_prox_list, cls_num)  # codes: [20, 72], dist: scalar
    vecs, dist = scipy.cluster.vq.vq(body_params_prox_list, codes)  # assign codes, vecs/dist: [1200]
    counts, bins = scipy.histogram(vecs, len(codes))  # count occurrences  count: [20]
    ee = entropy(counts)
    return ee, np.mean(dist)


if __name__ == '__main__':
    if args.dataset == 'prox':
        scene_list = ['MPH1Library', 'MPH16', 'N0SittingBooth', 'N3OpenArea']
    elif args.dataset == 'mp3d':
        scene_list = ['17DRP5sb8fy-bedroom', '17DRP5sb8fy-familyroomlounge',
                      '17DRP5sb8fy-livingroom', 'sKLMLpTHeUy-familyname_0_1',
                      'X7HyMhZNoso-livingroom_0_16', 'zsNo4HB9uLZ-bedroom0_0',
                      'zsNo4HB9uLZ-livingroom0_13']
    elif args.dataset == 'replica':
        scene_list = ['office_2', 'hotel_0', 'room_0', 'frl_apartment_0', 'apartment_1']


    body_params_prox_list_s2 = []
    cls_num = 20
    for scene in scene_list:
        body_params_prox_list_s2 += optimize_visulize(scene)

    entropy, dist = diversity(body_params_prox_list_s2, cls_num=cls_num)
    print('entropy:', entropy)
    print('distance:', dist)



