import warnings
warnings.simplefilter("ignore", UserWarning)

import argparse
import torch
from torch.utils import data
from tqdm import tqdm
from human_body_prior.tools.model_loader import load_vposer
import open3d as o3d
import smplx
import math
from utils import *
from utils_read_data import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--smplx_model_path', type=str,
                    default='/mnt/hdd/PROX/body_models/smplx_model',
                    help='path to smplx body model')
parser.add_argument('--vposer_model_path', type=str,
                    default='/mnt/hdd/PROX/body_models/vposer_v1_0',
                    help='path to vposer model')
parser.add_argument('--dataset', type=str, default='prox', help='choose dataset (prox/mp3d/replica)')
parser.add_argument('--dataset_path', type=str, default='/Users/siwei/Desktop/proxe', help='path to dataset')
parser.add_argument('--scene_name', type=str, default='N3OpenArea', help='scene name')
parser.add_argument('--optimize_result_dir', type=str, default='optimize_results/prox')
parser.add_argument("--visualize", default=False, type=bool, help='visualize scene/body mesh')
args = parser.parse_args()


# test scenes:
# prox:    ['MPH1Library', 'MPH16', 'N0SittingBooth', 'N3OpenArea']
# mp3d:    ['17DRP5sb8fy-bedroom', '17DRP5sb8fy-familyroomlounge',
#           '17DRP5sb8fy-livingroom', 'sKLMLpTHeUy-familyname_0_1',
#           'X7HyMhZNoso-livingroom_0_16', 'zsNo4HB9uLZ-bedroom0_0',
#           'zsNo4HB9uLZ-livingroom0_13']
# replica: ['office_2', 'hotel_0', 'room_0', 'frl_apartment_0', 'apartment_1']


def optimize_visulize():
    # read scene mesh, scene sdf
    scene, cur_scene_verts, s_grid_min_batch, s_grid_max_batch, s_sdf_batch = read_mesh_sdf(args.dataset_path,
                                                                                            args.dataset,
                                                                                            args.scene_name)
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

    vposer_model, _ = load_vposer(args.vposer_model_path, vp_model='snapshot')
    vposer_model = vposer_model.to(device)
    print('[INFO] vposer model loaded')


    ##################### load optimization results ##################
    shift_list = np.load('{}/{}/shift_list.npy'.format(args.optimize_result_dir, args.scene_name))
    rot_angle_list_1 = np.load('{}/{}/rot_angle_list_1.npy'.format(args.optimize_result_dir, args.scene_name))

    if args.optimize:
        body_params_opt_list_s1 = np.load('{}/{}/body_params_opt_list_s1.npy'.format(args.optimize_result_dir, args.scene_name))
        body_params_opt_list_s2 = np.load('{}/{}/body_params_opt_list_s2.npy'.format(args.optimize_result_dir, args.scene_name))
    body_verts_sample_list = np.load('{}/{}/body_verts_sample_list.npy'.format(args.optimize_result_dir, args.scene_name))
    n_sample = len(body_verts_sample_list)


    ########################## evaluation (contact/collision score) #########################
    loss_non_collision_sample, loss_contact_sample = 0, 0
    loss_non_collision_opt_s1, loss_contact_opt_s1 = 0, 0
    loss_non_collision_opt_s2, loss_contact_opt_s2 = 0, 0
    body_params_prox_list_s1, body_params_prox_list_s2 = [], []
    body_verts_opt_prox_s2_list = []

    for cnt in tqdm(range(0, n_sample)):
        body_verts_sample = body_verts_sample_list[cnt]  # [10475, 3]

        # smplx params --> body mesh
        body_params_opt_s1 = torch.from_numpy(body_params_opt_list_s1[cnt]).float().unsqueeze(0).to(device)  # [1,75]
        body_params_opt_s1 = convert_to_3D_rot(body_params_opt_s1)  # tensor, [bs=1, 72]
        body_pose_joint = vposer_model.decode(body_params_opt_s1[:, 16:48], output_type='aa').view(1,-1)  # [1, 63]
        body_verts_opt_s1 = gen_body_mesh(body_params_opt_s1, body_pose_joint, smplx_model)[0]  # [n_body_vert, 3]
        body_verts_opt_s1 = body_verts_opt_s1.detach().cpu().numpy()

        body_params_opt_s2 = torch.from_numpy(body_params_opt_list_s2[cnt]).float().unsqueeze(0).to(device)
        body_params_opt_s2 = convert_to_3D_rot(body_params_opt_s2)  # tensor, [bs=1, 72]
        body_pose_joint = vposer_model.decode(body_params_opt_s2[:, 16:48], output_type='aa').view(1, -1)
        body_verts_opt_s2 = gen_body_mesh(body_params_opt_s2, body_pose_joint, smplx_model)[0]
        body_verts_opt_s2 = body_verts_opt_s2.detach().cpu().numpy()

        ####################### transfrom local body verts to prox coodinate system ####################
        # generated body verts from cvae, before optimization
        body_verts_sample_prox = np.zeros(body_verts_sample.shape)   # [10475, 3]
        temp = body_verts_sample - shift_list[cnt]
        body_verts_sample_prox[:, 0] = temp[:, 0] * math.cos(-rot_angle_list_1[cnt]) - \
                                       temp[:, 1] * math.sin(-rot_angle_list_1[cnt])
        body_verts_sample_prox[:, 1] = temp[:, 0] * math.sin(-rot_angle_list_1[cnt]) + \
                                       temp[:, 1] * math.cos(-rot_angle_list_1[cnt])
        body_verts_sample_prox[:, 2] = temp[:, 2]

        ######### optimized body verts
        trans_matrix_1 = np.array([[math.cos(-rot_angle_list_1[cnt]), -math.sin(-rot_angle_list_1[cnt]), 0, 0],
                                   [math.sin(-rot_angle_list_1[cnt]), math.cos(-rot_angle_list_1[cnt]), 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])
        trans_matrix_2 = np.array([[1, 0, 0, -shift_list[cnt][0]],
                                   [0, 1, 0, -shift_list[cnt][1]],
                                   [0, 0, 1, -shift_list[cnt][2]],
                                   [0, 0, 0, 1]])
        ### stage 1: simple optimization results
        body_verts_opt_prox_s1 = np.zeros(body_verts_opt_s1.shape)  # [10475, 3]
        temp = body_verts_opt_s1 - shift_list[cnt]
        body_verts_opt_prox_s1[:, 0] = temp[:, 0] * math.cos(-rot_angle_list_1[cnt]) - \
                                       temp[:, 1] * math.sin(-rot_angle_list_1[cnt])
        body_verts_opt_prox_s1[:, 1] = temp[:, 0] * math.sin(-rot_angle_list_1[cnt]) + \
                                       temp[:, 1] * math.cos(-rot_angle_list_1[cnt])
        body_verts_opt_prox_s1[:, 2] = temp[:, 2]
        # transfrom local params to prox coordinate system
        body_params_prox_s1 = update_globalRT_for_smplx(body_params_opt_s1[0].cpu().numpy(), smplx_model, trans_matrix_2)  # [72]
        body_params_prox_s1 = update_globalRT_for_smplx(body_params_prox_s1, smplx_model, trans_matrix_1)  # [72]
        body_params_prox_list_s1.append(body_params_prox_s1)

        ### stage 2: advanced optimiation results
        body_verts_opt_prox_s2 = np.zeros(body_verts_opt_s2.shape)  # [10475, 3]
        temp = body_verts_opt_s2 - shift_list[cnt]
        body_verts_opt_prox_s2[:, 0] = temp[:, 0] * math.cos(-rot_angle_list_1[cnt]) - \
                                       temp[:, 1] * math.sin(-rot_angle_list_1[cnt])
        body_verts_opt_prox_s2[:, 1] = temp[:, 0] * math.sin(-rot_angle_list_1[cnt]) + \
                                       temp[:, 1] * math.cos(-rot_angle_list_1[cnt])
        body_verts_opt_prox_s2[:, 2] = temp[:, 2]
        # transfrom local params to prox coordinate system
        body_params_prox_s2 = update_globalRT_for_smplx(body_params_opt_s2[0].cpu().numpy(), smplx_model, trans_matrix_2)  # [72]
        body_params_prox_s2 = update_globalRT_for_smplx(body_params_prox_s2, smplx_model, trans_matrix_1)  # [72]
        body_params_prox_list_s2.append(body_params_prox_s2)
        body_verts_opt_prox_s2_list.append(body_verts_opt_prox_s2)

        ########################### visualization ##########################
        if args.visualize:
            body_mesh_sample = o3d.geometry.TriangleMesh()
            body_mesh_sample.vertices = o3d.utility.Vector3dVector(body_verts_sample_prox)
            body_mesh_sample.triangles = o3d.utility.Vector3iVector(smplx_model.faces)
            body_mesh_sample.compute_vertex_normals()

            body_mesh_opt_s1 = o3d.geometry.TriangleMesh()
            body_mesh_opt_s1.vertices = o3d.utility.Vector3dVector(body_verts_opt_prox_s1)
            body_mesh_opt_s1.triangles = o3d.utility.Vector3iVector(smplx_model.faces)
            body_mesh_opt_s1.compute_vertex_normals()

            body_mesh_opt_s2 = o3d.geometry.TriangleMesh()
            body_mesh_opt_s2.vertices = o3d.utility.Vector3dVector(body_verts_opt_prox_s2)
            body_mesh_opt_s2.triangles = o3d.utility.Vector3iVector(smplx_model.faces)
            body_mesh_opt_s2.compute_vertex_normals()

            o3d.visualization.draw_geometries([scene, body_mesh_sample])  # generated body mesh by cvae
            o3d.visualization.draw_geometries([scene, body_mesh_opt_s1])  # simple-optimized body mesh
            o3d.visualization.draw_geometries([scene, body_mesh_opt_s2])  # adv-optimizaed body mesh


        #####################  compute non-collision/contact score ##############
        # body verts before optimization
        body_verts_sample_prox_tensor = torch.from_numpy(body_verts_sample_prox).float().unsqueeze(0).to(device)  # [1, 10475, 3]
        norm_verts_batch = (body_verts_sample_prox_tensor - s_grid_min_batch) / (s_grid_max_batch - s_grid_min_batch) * 2 - 1
        body_sdf_batch = F.grid_sample(s_sdf_batch.unsqueeze(1),
                                       norm_verts_batch[:, :, [2, 1, 0]].view(-1, 10475, 1, 1, 3),
                                       padding_mode='border')
        if body_sdf_batch.lt(0).sum().item() < 1:  # if no interpenetration: negative sdf entries is less than one
            loss_non_collision_sample += 1.0
            loss_contact_sample += 0.0
        else:
            loss_non_collision_sample += (body_sdf_batch > 0).sum().float().item() / 10475.0
            loss_contact_sample += 1.0

        # stage 1: simple optimization results
        body_verts_opt_prox_tensor = torch.from_numpy(body_verts_opt_prox_s1).float().unsqueeze(0).to(device)  # [1, 10475, 3]
        norm_verts_batch = (body_verts_opt_prox_tensor - s_grid_min_batch) / (s_grid_max_batch - s_grid_min_batch) * 2 - 1
        body_sdf_batch = F.grid_sample(s_sdf_batch.unsqueeze(1),
                                       norm_verts_batch[:, :, [2, 1, 0]].view(-1, 10475, 1, 1, 3),
                                       padding_mode='border')
        if body_sdf_batch.lt(0).sum().item() < 1:  # if no interpenetration: negative sdf entries is less than one
            loss_non_collision_opt_s1 += 1.0
            loss_contact_opt_s1 += 0.0
        else:
            loss_non_collision_opt_s1 += (body_sdf_batch > 0).sum().float().item() / 10475.0
            loss_contact_opt_s1 += 1.0

        # stage 2: advanced optimization results
        body_verts_opt_prox_tensor = torch.from_numpy(body_verts_opt_prox_s2).float().unsqueeze(0).to(device)  # [1, 10475, 3]
        norm_verts_batch = (body_verts_opt_prox_tensor - s_grid_min_batch) / (s_grid_max_batch - s_grid_min_batch) * 2 - 1
        body_sdf_batch = F.grid_sample(s_sdf_batch.unsqueeze(1),
                                       norm_verts_batch[:, :, [2, 1, 0]].view(-1, 10475, 1, 1, 3),
                                       padding_mode='border')
        if body_sdf_batch.lt(0).sum().item() < 1:  # if no interpenetration: negative sdf entries is less than one
            loss_non_collision_opt_s2 += 1.0
            loss_contact_opt_s2 += 0.0
        else:
            loss_non_collision_opt_s2 += (body_sdf_batch > 0).sum().float().item() / 10475.0
            loss_contact_opt_s2 += 1.0


    print('scene:', args.scene_name)

    loss_non_collision_sample = loss_non_collision_sample / n_sample
    loss_contact_sample = loss_contact_sample / n_sample
    print('w/o optimization body: non_collision score:', loss_non_collision_sample)
    print('w/o optimization body: contact score:', loss_contact_sample)

    loss_non_collision_opt_s1 = loss_non_collision_opt_s1 / n_sample
    loss_contact_opt_s1 = loss_contact_opt_s1 / n_sample
    print('optimized body s1: non_collision score:', loss_non_collision_opt_s1)
    print('optimized body s1: contact score:', loss_contact_opt_s1)

    loss_non_collision_opt_s2 = loss_non_collision_opt_s2 / n_sample
    loss_contact_opt_s2 = loss_contact_opt_s2 / n_sample
    print('optimized body s2: non_collision score:', loss_non_collision_opt_s2)
    print('optimized body s2: contact score:', loss_contact_opt_s2)









if __name__ == '__main__':
    optimize_visulize()


