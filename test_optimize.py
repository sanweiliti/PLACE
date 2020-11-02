import warnings
warnings.simplefilter("ignore", UserWarning)

import argparse
import torch
from torch.utils import data
import torch.optim as optim
from tqdm import tqdm
from human_body_prior.tools.model_loader import load_vposer
import open3d as o3d
import smplx
from sklearn.neighbors import NearestNeighbors
import chamfer_pytorch.dist_chamfer as ext

from models.cvae import *
from loader.test_loader import TestLoader
from preprocess.preprocess_optimize import *
from preprocess.bps_encoding import *
from utils import *
from utils_read_data import *


parser = argparse.ArgumentParser()
parser.add_argument("--num_bps", default=10000, type=int, help='# of basis points')
parser.add_argument("--cube_size", default=2.0, type=float, help='size of 3D cage')
parser.add_argument("--eps_d", default=32, type=int, help='dimension of latent z')
# load trained models
parser.add_argument('--scene_bps_AE_path', type=str, default='checkpoints/sceneBpsAE_last_model.pkl')
parser.add_argument('--cVAE_path', type=str, default='checkpoints/cVAE_last_model.pkl')
parser.add_argument('--scene_verts_AE_path', type=str, default='checkpoints/sceneBpsVertsAE_last_model.pkl')
parser.add_argument('--bodyDec_path', type=str, default='checkpoints/body_dec_last_model.pkl')  # 38554
parser.add_argument('--smplx_model_path', type=str,
                    default='/Users/siwei/Desktop/body_models/smplx_model',
                    help='path to smplx body model')
parser.add_argument('--vposer_model_path', type=str,
                    default='/Users/siwei/Desktop/body_models/vposer_v1_0',
                    help='path to vposer model')
parser.add_argument('--dataset', type=str, default='prox', help='choose dataset (prox/mp3d/replica)')
parser.add_argument('--dataset_path', type=str, default='/mnt/hdd/PROX', help='path to dataset')
parser.add_argument('--scene_name', type=str, default='N3OpenArea', help='scene name')
parser.add_argument('--save_path', type=str,
                    default='optimize_results',
                    help='path to save optimized results')
parser.add_argument('--prox_dataset_path', type=str,
                    default='/mnt/hdd/PROX',
                    help='path to prox dataset, to load contact part index')
# optimize hype-parameters
parser.add_argument('--optimize', action='store_true', help='optimize or not')
parser.add_argument('--weight_loss_rec_verts', type=float, default=1.0,
                    help='weight for body vertex reconstruction loss')
parser.add_argument('--weight_loss_rec_bps', type=float, default=3.0,
                    help='weight for body bps feature reconstruction loss')
parser.add_argument('--weight_loss_vposer', type=float, default=0.02, help='weight for vpose loss')
parser.add_argument('--weight_loss_shape', type=float, default=0.01, help='weight for body shape prior loss')
parser.add_argument('--weight_loss_hand', type=float, default=0.01, help='weight for hand pose prior loss')
parser.add_argument('--weight_collision', type=float, default=8.0, help='weight for collision loss')
parser.add_argument('--weight_loss_contact', type=float, default=0.5, help='weight for contact loss')
parser.add_argument("--itr_s1", default=200, type=int, help='iterations for stage 1 (simple optimization)')
parser.add_argument("--itr_s2", default=100, type=int, help='iterations for stage 2 (advanced optimization)')
parser.add_argument("--n_sample", default=1200, type=int, help='# of evaluation samples')
args = parser.parse_args()



# test scenes:
# prox:    ['MPH1Library', 'MPH16', 'N0SittingBooth', 'N3OpenArea']
# mp3d:    ['17DRP5sb8fy-bedroom', '17DRP5sb8fy-familyroomlounge',
#           '17DRP5sb8fy-livingroom', 'sKLMLpTHeUy-familyname_0_1',
#           'X7HyMhZNoso-livingroom_0_16', 'zsNo4HB9uLZ-bedroom0_0',
#           'zsNo4HB9uLZ-livingroom0_13']
# replica: ['office_2', 'hotel_0', 'room_0', 'frl_apartment_0', 'apartment_1']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimize():
    scene_mesh, cur_scene_verts, s_grid_min_batch, s_grid_max_batch, s_sdf_batch = read_mesh_sdf(args.dataset_path,
                                                                                                 args.dataset,
                                                                                                 args.scene_name)
    save_path = '{}/{}'.format(args.save_path, args.dataset, args.scene_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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


    ####################### calculate scene bps representation ##################
    rot_angle, scene_min_x, scene_max_x, scene_min_y, scene_max_y = define_scene_boundary(args.dataset, args.scene_name)
    
    scene_verts_crop_local_list, scene_verts_local_list, = [], []
    shift_list = []
    rot_angle_list_1, rot_angle_list_2 = [], []

    np.random.seed(0)
    random_seed_list = np.random.randint(10000, size=args.n_sample)
    for i in tqdm(range(args.n_sample)):
        scene_verts = rotate_scene_smplx_predefine(cur_scene_verts, rot_angle=rot_angle)
        scene_verts_local, scene_verts_crop_local, shift = crop_scene_cube_smplx_predifine(
            scene_verts, r=args.cube_size, with_wall_ceilling=True, random_seed=random_seed_list[i],
            scene_min_x=scene_min_x, scene_max_x=scene_max_x, scene_min_y=scene_min_y, scene_max_y=scene_max_y,
            rotate=True)
        scene_verts_crop_local_list.append(scene_verts_crop_local)  # list, different verts num for each cropped scene
        scene_verts_local_list.append(scene_verts_local)
        shift_list.append(shift)
        rot_angle_list_1.append(rot_angle)
    print('[INFO] scene mesh cropped and shifted.')


    scene_bps_list, body_bps_list = [], []
    scene_bps_verts_global_list, scene_bps_verts_local_list = [], []
    scene_bps_verts_global_list = []
    scene_basis_set = bps_gen_ball_inside(n_bps=args.num_bps, random_seed=100)
    np.random.seed(1)
    random_seed_list = np.random.randint(10000, size=args.n_sample)
    for i in tqdm(range(args.n_sample)):
        scene_verts_global, scene_verts_crop_global, rot_angle = \
            augmentation_crop_scene_smplx(scene_verts_local_list[i] / args.cube_size,
                                          scene_verts_crop_local_list[i] / args.cube_size,
                                          random_seed_list[i])
        scene_bps, selected_scene_verts_global, selected_ind = bps_encode_scene(scene_basis_set,
                                                                                scene_verts_crop_global)  # [n_feat, n_bps]
        scene_bps_list.append(scene_bps)
        selected_scene_verts_local = scene_verts_crop_local_list[i][selected_ind]
        scene_bps_verts_local_list.append(selected_scene_verts_local)
        scene_bps_verts_global_list.append(selected_scene_verts_global)
        rot_angle_list_2.append(rot_angle)

    scene_bps_list = np.asarray(scene_bps_list)  # [n_sample*4, n_feat, n_bps]
    scene_bps_verts_local_list = np.asarray(scene_bps_verts_local_list)
    scene_verts_local_list = np.asarray(scene_verts_local_list)
    np.save('{}/scene_bps_list.npy'.format(save_path), scene_bps_list)
    print('[INFO] scene bps/verts saved.')


    ######################## set dataloader and load model ########################
    dataset = TestLoader()
    dataset.n_samples = args.n_sample
    dataset.scene_bps_list = scene_bps_list  # [n_sample, n_feat, n_bps]
    dataset.scene_bps_verts_list = scene_bps_verts_local_list
    print('[INFO] dataloader updated, select n_samples={}'.format(dataset.__len__()))
    test_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                                                  num_workers=0, drop_last=False)

    scene_bps_AE = BPSRecMLP(n_bps=args.num_bps, n_bps_feat=1, hsize1=1024, hsize2=512).to(device)
    weights = torch.load(args.scene_bps_AE_path, map_location=lambda storage, loc: storage)
    scene_bps_AE.load_state_dict(weights)

    c_VAE = BPS_CVAE(n_bps=args.num_bps, n_bps_feat=1, hsize1=1024, hsize2=512, eps_d=args.eps_d).to(device)
    weights = torch.load(args.cVAE_path, map_location=lambda storage, loc: storage)
    c_VAE.load_state_dict(weights)

    scene_AE = Verts_AE(n_bps=10000, hsize1=1024, hsize2=512).to(device)
    weights = torch.load(args.scene_verts_AE_path, map_location=lambda storage, loc: storage)
    scene_AE.load_state_dict(weights)

    body_dec = Body_Dec_shift(n_bps=10000, n_bps_feat=1, hsize1=1024, hsize2=512, n_body_verts=10475,
                              body_param_dim=75, rec_goal='body_verts').to(device)
    weights = torch.load(args.bodyDec_path, map_location=lambda storage, loc: storage)
    body_dec.load_state_dict(weights)

    scene_bps_AE.eval()
    c_VAE.eval()
    scene_AE.eval()
    body_dec.eval()


    ######################## initialize for optimization ##########################
    body_verts_sample_list = []
    body_bps_sample_list = []
    np.random.seed(2)
    random_seed_list = np.random.randint(10000, size=args.n_sample)
    with torch.no_grad():
        for step, data in tqdm(enumerate(test_dataloader)):
            [scene_bps, scene_bps_verts] = [item.to(device) for item in data]
            scene_bps_verts = scene_bps_verts / args.cube_size

            _, scene_bps_feat = scene_bps_AE(scene_bps)
            _, scene_bps_verts_feat = scene_AE(scene_bps_verts)
            # torch.manual_seed(random_seed_list[step])
            body_bps_sample = c_VAE.sample(1, scene_bps_feat)  # [1, 1, 10000]
            body_bps_sample_list.append(body_bps_sample[0].detach().cpu().numpy())  # [n, 1, 10000]
            body_verts_sample, body_shift = body_dec(body_bps_sample, scene_bps_verts_feat)  # [1, 3, 10475], unit ball scale, local coordinate

            # shifted generated body
            body_shift = body_shift.repeat(1, 1, 10475).reshape([body_verts_sample.shape[0], 10475, 3])  # [bs, 10475, 3]
            body_verts_sample = body_verts_sample + body_shift.permute(0, 2, 1)  # [bs, 3, 10475]

            body_verts_sample_list.append(body_verts_sample[0].detach().cpu().numpy())  # [n, 3, 10475]

    contact_part = ['L_Leg', 'R_Leg']
    vid, _ = get_contact_id(body_segments_folder=os.path.join(args.prox_dataset_path, 'body_segments'),
                            contact_body_parts=contact_part)


    ############################### save data ###############################
    shift_list = np.asarray(shift_list)
    rot_angle_list_1 = np.asarray(rot_angle_list_1)
    rot_angle_list_2 = np.asarray(rot_angle_list_2)
    body_bps_sample_list = np.asarray(body_bps_sample_list)
    body_verts_sample_list = np.asarray(body_verts_sample_list)

    np.save('{}/shift_list.npy'.format(save_path), shift_list)
    np.save('{}/rot_angle_list_1.npy'.format(save_path), rot_angle_list_1)
    np.save('{}/rot_angle_list_2.npy'.format(save_path), rot_angle_list_2)
    np.save('{}/body_bps_sample_list.npy'.format(save_path), body_bps_sample_list)
    # save generated body verts in original scale
    np.save('{}/body_verts_sample_list.npy'.format(save_path), body_verts_sample_list.transpose((0,2,1))*args.cube_size)


    ############################### optimization ##################################
    if args.optimize:
        #################### stage 1 (simple optimization, without contact/collision loss) ###################
        print('[INFO] start optimization stage 1...')
        body_params_opt_list_s1 = []
        for cnt in range(args.n_sample):
            print('stage 1: current cnt:', cnt)
            body_params_rec = torch.randn(1, 72).to(device)  # initiliza smplx params, bs=1, local coordinate system
            body_params_rec[0, 0] = 0.0
            body_params_rec[0, 1] = 0.0
            body_params_rec[0, 2] = 0.0
            body_params_rec[0, 3] = 1.5
            body_params_rec[0, 4] = 0.0
            body_params_rec[0, 5] = 0.0
            body_params_rec = convert_to_6D_rot(body_params_rec)
            body_params_rec.requires_grad = True

            optimizer = optim.Adam([body_params_rec], lr=0.1)

            body_bps = torch.from_numpy(body_bps_sample_list[cnt]).float().unsqueeze(0).to(device)  # [bs=1, 1, 10000]
            body_verts = torch.from_numpy(body_verts_sample_list[cnt]).float().unsqueeze(0).to(device)  # [bs=1, 3, 10475]
            body_verts = body_verts.permute(0, 2, 1)  # [1, 10475, 3]
            body_verts = body_verts * args.cube_size  # to local coordinate system scale

            for step in tqdm(range(args.itr_s1)):
                if step > 100:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 0.01
                if step > 300:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 0.001
                optimizer.zero_grad()

                body_params_rec_72 = convert_to_3D_rot(body_params_rec)  # tensor, [bs=1, 72]
                body_pose_joint = vposer_model.decode(body_params_rec_72[:, 16:48], output_type='aa').view(1, -1)  # tensor, [bs=1, 63]
                body_verts_rec = gen_body_mesh(body_params_rec_72, body_pose_joint, smplx_model)[0]  # tensor, [n_body_vert, 3]

                # transform body verts to unit ball global coordinate system
                temp = body_verts_rec / args.cube_size  # scale into unit ball
                body_verts_rec_global = torch.zeros(body_verts_rec.shape).to(device)
                body_verts_rec_global[:, 0] = temp[:, 0] * math.cos(rot_angle_list_2[cnt]) - \
                                              temp[:, 1] * math.sin(rot_angle_list_2[cnt])
                body_verts_rec_global[:, 1] = temp[:, 0] * math.sin(rot_angle_list_2[cnt]) + \
                                              temp[:, 1] * math.cos(rot_angle_list_2[cnt])
                body_verts_rec_global[:, 2] = temp[:, 2]

                # calculate optimized body bps feature
                body_bps_rec = torch.zeros(body_bps.shape)
                if args.weight_loss_rec_bps > 0:
                    nbrs = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="ball_tree").fit(body_verts_rec_global.detach().cpu().numpy())
                    neigh_dist, neigh_ind = nbrs.kneighbors(scene_bps_verts_global_list[cnt])
                    body_bps_rec = body_verts_rec_global[neigh_ind.squeeze()] - \
                                   torch.from_numpy(scene_bps_verts_global_list[cnt]).float().to(device)  # [n_bps, 3]
                    body_bps_rec = torch.sqrt(body_bps_rec[:, 0] ** 2 + body_bps_rec[:, 1] ** 2 + body_bps_rec[:, 2] ** 2).unsqueeze(0).unsqueeze(0)  # [bs=1, 1, n_bps]

                ### body bps feature reconstruct loss
                loss_rec_verts = F.l1_loss(body_verts_rec.unsqueeze(0), body_verts)
                loss_rec_bps = F.l1_loss(body_bps, body_bps_rec)

                ### vposer loss
                body_params_rec_72 = convert_to_3D_rot(body_params_rec)
                vposer_pose = body_params_rec_72[:, 16:48]
                loss_vposer = torch.mean(vposer_pose ** 2)
                ### shape prior loss
                shape_params = body_params_rec_72[:, 6:16]
                loss_shape = torch.mean(shape_params ** 2)
                ### hand pose prior loss
                hand_params = body_params_rec_72[:, 48:]
                loss_hand = torch.mean(hand_params ** 2)

                loss = args.weight_loss_rec_verts * loss_rec_verts + args.weight_loss_rec_bps * loss_rec_bps + \
                       args.weight_loss_vposer * loss_vposer + \
                       args.weight_loss_shape * loss_shape + \
                       args.weight_loss_hand * loss_hand
                loss.backward(retain_graph=True)
                optimizer.step()

            body_params_opt_list_s1.append(body_params_rec[0].detach().cpu().numpy())


        body_params_opt_list_s1 = np.asarray(body_params_opt_list_s1)
        np.save('{}/body_params_opt_list_s1.npy'.format(save_path), body_params_opt_list_s1)


        ################ stage 2 (advanced optimization, with contact/collision loss) ##################
        print('[INFO] start optimization stage 2...')
        body_params_opt_list_s2 = []
        for cnt in range(args.n_sample):
            print('current cnt:', cnt)
            body_params_rec = body_params_opt_list_s1[cnt]  # [75]
            body_params_rec = torch.from_numpy(body_params_rec).float().to(device).unsqueeze(0)
            body_params_rec.requires_grad = True

            optimizer = optim.Adam([body_params_rec], lr=0.01)

            body_bps = torch.from_numpy(body_bps_sample_list[cnt]).float().unsqueeze(0).to(device)  # [bs=1, 1, 10000]
            body_verts = torch.from_numpy(body_verts_sample_list[cnt]).float().unsqueeze(0).to(device)
            body_verts = body_verts.permute(0, 2, 1)  # [1, 10475, 3]
            body_verts = body_verts * args.cube_size  # to local coordinate system scale

            for step in tqdm(range(args.itr_s2)):
                optimizer.zero_grad()

                body_params_rec_72 = convert_to_3D_rot(body_params_rec)  # tensor, [bs=1, 72]
                body_pose_joint = vposer_model.decode(body_params_rec_72[:, 16:48], output_type='aa').view(1,-1)  # tensor, [bs=1, 63]
                body_verts_rec = gen_body_mesh(body_params_rec_72, body_pose_joint, smplx_model)[0]  # tensor, [n_body_vert, 3]

                # transform body verts to unit ball global coordinate
                temp = body_verts_rec / args.cube_size  # scale into unit ball
                body_verts_rec_global = torch.zeros(body_verts_rec.shape).to(device)
                body_verts_rec_global[:, 0] = temp[:, 0] * math.cos(rot_angle_list_2[cnt]) - \
                                              temp[:, 1] * math.sin(rot_angle_list_2[cnt])
                body_verts_rec_global[:, 1] = temp[:, 0] * math.sin(rot_angle_list_2[cnt]) + \
                                              temp[:, 1] * math.cos(rot_angle_list_2[cnt])
                body_verts_rec_global[:, 2] = temp[:, 2]

                # calculate body_bps_rec
                body_bps_rec = torch.zeros(body_bps.shape)
                if args.weight_loss_rec_bps > 0:
                    nbrs = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="ball_tree").fit(body_verts_rec_global.detach().cpu().numpy())
                    neigh_dist, neigh_ind = nbrs.kneighbors(scene_bps_verts_global_list[cnt])
                    body_bps_rec = body_verts_rec_global[neigh_ind.squeeze()] - \
                                   torch.from_numpy(scene_bps_verts_global_list[cnt]).float().to(device)  # [n_bps, 3]
                    body_bps_rec = torch.sqrt(body_bps_rec[:, 0] ** 2 + body_bps_rec[:, 1] ** 2 + body_bps_rec[:, 2] ** 2).unsqueeze(0).unsqueeze(0)  # [bs=1, 1, n_bps]

                ### body bps encoding reconstruct loss
                loss_rec_verts = F.l1_loss(body_verts_rec.unsqueeze(0), body_verts)
                loss_rec_bps = F.l1_loss(body_bps, body_bps_rec)

                ### vposer loss
                body_params_rec_72 = convert_to_3D_rot(body_params_rec)
                vposer_pose = body_params_rec_72[:, 16:48]
                loss_vposer = torch.mean(vposer_pose ** 2)
                ### shape prior loss
                shape_params = body_params_rec_72[:, 6:16]
                loss_shape = torch.mean(shape_params ** 2)
                ### hand pose prior loss
                hand_params = body_params_rec_72[:, 48:]
                loss_hand = torch.mean(hand_params ** 2)

                # transfrom local body_verts_rec to prox coordinate system
                body_verts_rec_prox = torch.zeros(body_verts_rec.shape).to(device)
                temp = body_verts_rec - torch.from_numpy(shift_list[cnt]).float().to(device)
                body_verts_rec_prox[:, 0] = temp[:, 0] * math.cos(-rot_angle_list_1[cnt]) - \
                                            temp[:, 1] * math.sin(-rot_angle_list_1[cnt])
                body_verts_rec_prox[:, 1] = temp[:, 0] * math.sin(-rot_angle_list_1[cnt]) + \
                                            temp[:, 1] * math.cos(-rot_angle_list_1[cnt])
                body_verts_rec_prox[:, 2] = temp[:, 2]
                body_verts_rec_prox = body_verts_rec_prox.unsqueeze(0)  # tensor, [bs=1, 10475, 3]

                ### sdf collision loss
                norm_verts_batch = (body_verts_rec_prox - s_grid_min_batch) / (s_grid_max_batch - s_grid_min_batch) * 2 - 1
                n_verts = norm_verts_batch.shape[1]
                body_sdf_batch = F.grid_sample(s_sdf_batch.unsqueeze(1),
                                               norm_verts_batch[:, :, [2, 1, 0]].view(-1, n_verts, 1, 1, 3),
                                               padding_mode='border')
                # if there are no penetrating vertices then set sdf_penetration_loss = 0
                if body_sdf_batch.lt(0).sum().item() < 1:
                    loss_collision = torch.tensor(0.0, dtype=torch.float32).to(device)
                else:
                    loss_collision = body_sdf_batch[body_sdf_batch < 0].abs().mean()

                ### contact loss
                body_verts_contact = body_verts_rec.unsqueeze(0)[:, vid, :]  # [1,1121,3]
                dist_chamfer_contact = ext.chamferDist()
                # scene_verts: [bs=1, n_scene_verts, 3]
                scene_verts = torch.from_numpy(scene_verts_local_list[cnt]).float().to(device).unsqueeze(0)  # [1,50000,3]
                contact_dist, _ = dist_chamfer_contact(body_verts_contact.contiguous(),
                                                       scene_verts.contiguous())
                loss_contact = torch.mean(torch.sqrt(contact_dist + 1e-4) / (torch.sqrt(contact_dist + 1e-4) + 1.0))

                loss = args.weight_loss_rec_verts * loss_rec_verts + args.weight_loss_rec_bps * loss_rec_bps + \
                       args.weight_loss_vposer * loss_vposer + \
                       args.weight_loss_shape * loss_shape + \
                       args.weight_loss_hand * loss_hand + \
                       args.weight_collision * loss_collision + args.weight_loss_contact * loss_contact
                loss.backward(retain_graph=True)
                optimizer.step()

            body_params_opt_list_s2.append(body_params_rec[0].detach().cpu().numpy())

        body_params_opt_list_s2 = np.asarray(body_params_opt_list_s2)
        np.save('{}/body_params_opt_list_s2.npy'.format(save_path), body_params_opt_list_s2)




if __name__ == '__main__':
    optimize()