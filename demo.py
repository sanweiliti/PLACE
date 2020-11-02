import warnings
warnings.simplefilter("ignore", UserWarning)

# from open3d import JVisualizer
from open3d.j_visualizer import JVisualizer
import torch
import torch.optim as optim
from tqdm import tqdm
from human_body_prior.tools.model_loader import load_vposer
import open3d as o3d
import smplx
from sklearn.neighbors import NearestNeighbors
import chamfer_pytorch.dist_chamfer as ext
from models.cvae import *
from preprocess.preprocess_optimize import *
from preprocess.bps_encoding import *
from utils import *
from utils_read_data import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimize():
    prox_dataset_path = '/mnt/hdd/PROX'
    scene_name = 'N3OpenArea'

    # set optimization hype-parameters
    weight_loss_rec_verts = 1.0
    weight_loss_rec_bps = 3.0
    weight_loss_vposer = 0.02
    weight_loss_shape = 0.01
    weight_loss_hand = 0.01
    weight_collision = 8.0
    weight_loss_contact = 0.5
    itr_s1 = 200
    itr_s2 = 100

    cube_size = 2.0  # 3D cage size
    optimize = True  # optimize or not
    # train model path
    scene_bps_AE_path = 'checkpoints/sceneBpsAE_last_model.pkl'
    cVAE_path = 'checkpoints/cVAE_last_model.pkl'
    scene_verts_AE_path = 'checkpoints/sceneBpsVertsAE_last_model.pkl'
    bodyDec_path = 'checkpoints/body_dec_last_model.pkl'
    # smplx/vpose model path
    smplx_model_path = '/mnt/hdd/PROX/body_models/smplx_model'
    vposer_model_path = '/mnt/hdd/PROX/body_models/vposer_v1_0'

    # read scen mesh/sdf
    scene_mesh, cur_scene_verts, s_grid_min_batch, s_grid_max_batch, s_sdf_batch = read_mesh_sdf(prox_dataset_path,
                                                                                                 'prox',
                                                                                                 scene_name)
    smplx_model = smplx.create(smplx_model_path, model_type='smplx',
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

    vposer_model, _ = load_vposer(vposer_model_path, vp_model='snapshot')
    vposer_model = vposer_model.to(device)
    print('[INFO] vposer model loaded')


    ####################### calculate scene bps representation ##################
    rot_angle_1, scene_min_x, scene_max_x, scene_min_y, scene_max_y = define_scene_boundary('prox', scene_name)

    scene_verts = rotate_scene_smplx_predefine(cur_scene_verts, rot_angle=rot_angle_1)
    scene_verts_local, scene_verts_crop_local, shift = crop_scene_cube_smplx_predifine(
        scene_verts, r=cube_size, with_wall_ceilling=True, random_seed=np.random.randint(10000),
        scene_min_x=scene_min_x, scene_max_x=scene_max_x, scene_min_y=scene_min_y, scene_max_y=scene_max_y,
        rotate=True)
    print('[INFO] scene mesh cropped and shifted.')


    scene_basis_set = bps_gen_ball_inside(n_bps=10000, random_seed=100)
    scene_verts_global, scene_verts_crop_global, rot_angle_2 = \
        augmentation_crop_scene_smplx(scene_verts_local / cube_size,
                                      scene_verts_crop_local / cube_size,
                                      np.random.randint(10000))
    scene_bps, selected_scene_verts_global, selected_ind = bps_encode_scene(scene_basis_set,
                                                                            scene_verts_crop_global)  # [n_feat, n_bps]
    selected_scene_verts_local = scene_verts_crop_local[selected_ind]



    ############################# load trained model ###############################
    scene_bps = torch.from_numpy(scene_bps).float().unsqueeze(0).to(device)  # [1, 1, n_bps]
    scene_bps_verts = torch.from_numpy(selected_scene_verts_local.transpose(1, 0)).float().unsqueeze(0).to(device)  # [1, 3, 10000]

    scene_bps_AE = BPSRecMLP(n_bps=10000, n_bps_feat=1, hsize1=1024, hsize2=512).to(device)
    weights = torch.load(scene_bps_AE_path, map_location=lambda storage, loc: storage)
    scene_bps_AE.load_state_dict(weights)

    c_VAE = BPS_CVAE(n_bps=10000, n_bps_feat=1, hsize1=1024, hsize2=512, eps_d=32).to(device)
    weights = torch.load(cVAE_path, map_location=lambda storage, loc: storage)
    c_VAE.load_state_dict(weights)

    scene_AE = Verts_AE(n_bps=10000, hsize1=1024, hsize2=512).to(device)
    weights = torch.load(scene_verts_AE_path, map_location=lambda storage, loc: storage)
    scene_AE.load_state_dict(weights)

    body_dec = Body_Dec_shift(n_bps=10000, n_bps_feat=1, hsize1=1024, hsize2=512, n_body_verts=10475,
                              body_param_dim=75, rec_goal='body_verts').to(device)
    weights = torch.load(bodyDec_path, map_location=lambda storage, loc: storage)
    body_dec.load_state_dict(weights)

    scene_bps_AE.eval()
    c_VAE.eval()
    scene_AE.eval()
    body_dec.eval()

    ######################## sample a body / initialize for optimization ##########################
    scene_bps_verts = scene_bps_verts / cube_size

    _, scene_bps_feat = scene_bps_AE(scene_bps)
    _, scene_bps_verts_feat = scene_AE(scene_bps_verts)
    body_bps_sample = c_VAE.sample(1, scene_bps_feat)  # [1, 1, 10000]
    body_verts_sample, body_shift = body_dec(body_bps_sample,
                                             scene_bps_verts_feat)  # [1, 3, 10475], unit ball scale, local coordinate
    body_shift = body_shift.repeat(1, 1, 10475).reshape([body_verts_sample.shape[0], 10475, 3])  # [bs, 10475, 3]
    body_verts_sample = body_verts_sample + body_shift.permute(0, 2, 1)  # [bs=1, 3, 10475]

    # visualize generated body
    body_verts_sample_prox = np.zeros([10475, 3])  # [10475, 3]
    temp = body_verts_sample[0].detach().cpu().numpy().transpose((1,0)) * cube_size - shift
    body_verts_sample_prox[:, 0] = temp[:, 0] * math.cos(-rot_angle_1) - \
                                   temp[:, 1] * math.sin(-rot_angle_1)
    body_verts_sample_prox[:, 1] = temp[:, 0] * math.sin(-rot_angle_1) + \
                                   temp[:, 1] * math.cos(-rot_angle_1)
    body_verts_sample_prox[:, 2] = temp[:, 2]

    body_mesh = o3d.geometry.TriangleMesh()
    body_mesh.vertices = o3d.utility.Vector3dVector(body_verts_sample_prox)
    body_mesh.triangles = o3d.utility.Vector3iVector(smplx_model.faces)
    body_mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([scene_mesh, body_mesh])


    ######################################## optimization ##################################
    if optimize:
        contact_part = ['L_Leg', 'R_Leg']
        vid, _ = get_contact_id(body_segments_folder=os.path.join(prox_dataset_path, 'body_segments'),
                                contact_body_parts=contact_part)

        ################ stage 1 (simple optimization, without contact/collision loss) ######
        print('[INFO] start optimization stage 1...')
        body_params_rec = torch.randn(1, 72).to(device)  # initialize smplx params, bs=1, local 3D cage coordinate system
        body_params_rec[0, 0] = 0.0
        body_params_rec[0, 1] = 0.0
        body_params_rec[0, 2] = 0.0
        body_params_rec[0, 3] = 1.5
        body_params_rec[0, 4] = 0.0
        body_params_rec[0, 5] = 0.0
        body_params_rec = convert_to_6D_rot(body_params_rec)
        body_params_rec.requires_grad = True
        optimizer = optim.Adam([body_params_rec], lr=0.1)

        body_verts = body_verts_sample.permute(0, 2, 1)  # [1, 10475, 3]
        body_verts = body_verts * cube_size  # to local 3d cage coordinate system scale

        for step in tqdm(range(itr_s1)):
            if step > 100:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.01
            if step > 300:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.001
            optimizer.zero_grad()

            body_params_rec_72 = convert_to_3D_rot(body_params_rec)  # tensor, [bs=1, 72]
            body_pose_joint = vposer_model.decode(body_params_rec_72[:, 16:48], output_type='aa').view(1, -1)
            body_verts_rec = gen_body_mesh(body_params_rec_72, body_pose_joint, smplx_model)[0]  # [n_body_vert, 3]

            # transform body verts to unit ball global coordinate system
            temp = body_verts_rec / cube_size  # scale into unit ball
            body_verts_rec_global = torch.zeros(body_verts_rec.shape).to(device)
            body_verts_rec_global[:, 0] = temp[:, 0] * math.cos(rot_angle_2) - \
                                          temp[:, 1] * math.sin(rot_angle_2)
            body_verts_rec_global[:, 1] = temp[:, 0] * math.sin(rot_angle_2) + \
                                          temp[:, 1] * math.cos(rot_angle_2)
            body_verts_rec_global[:, 2] = temp[:, 2]

            # calculate optimized body bps feature
            body_bps_rec = torch.zeros(body_bps_sample.shape)
            if weight_loss_rec_bps > 0:
                nbrs = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="ball_tree").fit(
                    body_verts_rec_global.detach().cpu().numpy())
                neigh_dist, neigh_ind = nbrs.kneighbors(selected_scene_verts_global)
                body_bps_rec = body_verts_rec_global[neigh_ind.squeeze()] - \
                               torch.from_numpy(selected_scene_verts_global).float().to(device)  # [n_bps, 3]
                body_bps_rec = torch.sqrt(
                    body_bps_rec[:, 0] ** 2 + body_bps_rec[:, 1] ** 2 + body_bps_rec[:, 2] ** 2).unsqueeze(0).unsqueeze(0)  # [bs=1, 1, n_bps]

            ### body bps feature reconstruct loss
            loss_rec_verts = F.l1_loss(body_verts_rec.unsqueeze(0), body_verts)
            loss_rec_bps = F.l1_loss(body_bps_sample, body_bps_rec)

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

            loss = weight_loss_rec_verts * loss_rec_verts + weight_loss_rec_bps * loss_rec_bps + \
                   weight_loss_vposer * loss_vposer + \
                   weight_loss_shape * loss_shape + \
                   weight_loss_hand * loss_hand
            loss.backward(retain_graph=True)
            optimizer.step()

        print('[INFO] optimization stage 1 finished.')

        # # visualize body mesh after simple optimization (stage 1)
        # # smplx params --> body mesh
        # body_params_opt_s1 = convert_to_3D_rot(body_params_rec)  # tensor, [bs=1, 72]
        # body_pose_joint = vposer_model.decode(body_params_opt_s1[:, 16:48], output_type='aa').view(1, -1)
        # body_verts_opt_s1 = gen_body_mesh(body_params_opt_s1, body_pose_joint, smplx_model)[0]
        # body_verts_opt_s1 = body_verts_opt_s1.detach().cpu().numpy()  # [n_body_vert, 3]
        #
        # body_verts_opt_prox_s1 = np.zeros(body_verts_opt_s1.shape)  # [10475, 3]
        # temp = body_verts_opt_s1 - shift
        # body_verts_opt_prox_s1[:, 0] = temp[:, 0] * math.cos(-rot_angle_1) - \
        #                                temp[:, 1] * math.sin(-rot_angle_1)
        # body_verts_opt_prox_s1[:, 1] = temp[:, 0] * math.sin(-rot_angle_1) + \
        #                                temp[:, 1] * math.cos(-rot_angle_1)
        # body_verts_opt_prox_s1[:, 2] = temp[:, 2]
        #
        # body_mesh_opt_s1 = o3d.geometry.TriangleMesh()
        # body_mesh_opt_s1.vertices = o3d.utility.Vector3dVector(body_verts_opt_prox_s1)
        # body_mesh_opt_s1.triangles = o3d.utility.Vector3iVector(smplx_model.faces)
        # body_mesh_opt_s1.compute_vertex_normals()
        # o3d.visualization.draw_geometries([scene_mesh, body_mesh_opt_s1])


        ################ stage 2 (advanced optimization, with contact/collision loss) ##################
        print('[INFO] start optimization stage 2...')
        optimizer = optim.Adam([body_params_rec], lr=0.01)

        body_verts = body_verts_sample.permute(0, 2, 1)  # [1, 10475, 3]
        body_verts = body_verts * cube_size  # to local 3d cage coordinate system scale

        for step in tqdm(range(itr_s2)):
            optimizer.zero_grad()

            body_params_rec_72 = convert_to_3D_rot(body_params_rec)  # tensor, [bs=1, 72]
            body_pose_joint = vposer_model.decode(body_params_rec_72[:, 16:48], output_type='aa').view(1,-1)
            body_verts_rec = gen_body_mesh(body_params_rec_72, body_pose_joint, smplx_model)[0]  # [n_body_vert, 3]

            # transform body verts to unit ball global coordinate
            temp = body_verts_rec / cube_size  # scale into unit ball
            body_verts_rec_global = torch.zeros(body_verts_rec.shape).to(device)
            body_verts_rec_global[:, 0] = temp[:, 0] * math.cos(rot_angle_2) - \
                                          temp[:, 1] * math.sin(rot_angle_2)
            body_verts_rec_global[:, 1] = temp[:, 0] * math.sin(rot_angle_2) + \
                                          temp[:, 1] * math.cos(rot_angle_2)
            body_verts_rec_global[:, 2] = temp[:, 2]

            # calculate body_bps_rec
            body_bps_rec = torch.zeros(body_bps_sample.shape)
            if weight_loss_rec_bps > 0:
                nbrs = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="ball_tree").fit(
                    body_verts_rec_global.detach().cpu().numpy())
                neigh_dist, neigh_ind = nbrs.kneighbors(selected_scene_verts_global)
                body_bps_rec = body_verts_rec_global[neigh_ind.squeeze()] - \
                               torch.from_numpy(selected_scene_verts_global).float().to(device)  # [n_bps, 3]
                body_bps_rec = torch.sqrt(
                    body_bps_rec[:, 0] ** 2 + body_bps_rec[:, 1] ** 2 + body_bps_rec[:, 2] ** 2).unsqueeze(0).unsqueeze(0)  # [bs=1, 1, n_bps]

            ### body bps encoding reconstruct loss
            loss_rec_verts = F.l1_loss(body_verts_rec.unsqueeze(0), body_verts)
            loss_rec_bps = F.l1_loss(body_bps_sample, body_bps_rec)

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

            # transfrom body_verts_rec (local 3d cage coordinate system) to prox coordinate system
            body_verts_rec_prox = torch.zeros(body_verts_rec.shape).to(device)
            temp = body_verts_rec - torch.from_numpy(shift).float().to(device)
            body_verts_rec_prox[:, 0] = temp[:, 0] * math.cos(-rot_angle_1) - \
                                        temp[:, 1] * math.sin(-rot_angle_1)
            body_verts_rec_prox[:, 1] = temp[:, 0] * math.sin(-rot_angle_1) + \
                                        temp[:, 1] * math.cos(-rot_angle_1)
            body_verts_rec_prox[:, 2] = temp[:, 2]
            body_verts_rec_prox = body_verts_rec_prox.unsqueeze(0)  # tensor, [bs=1, 10475, 3]

            ### sdf collision loss
            norm_verts_batch = (body_verts_rec_prox - s_grid_min_batch) / (
                    s_grid_max_batch - s_grid_min_batch) * 2 - 1
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
            scene_verts = torch.from_numpy(scene_verts_local).float().to(device).unsqueeze(0)  # [1,50000,3]
            contact_dist, _ = dist_chamfer_contact(body_verts_contact.contiguous(),
                                                   scene_verts.contiguous())
            loss_contact = torch.mean(torch.sqrt(contact_dist + 1e-4) / (torch.sqrt(contact_dist + 1e-4) + 1.0))

            loss = weight_loss_rec_verts * loss_rec_verts + weight_loss_rec_bps * loss_rec_bps + \
                   weight_loss_vposer * loss_vposer + \
                   weight_loss_shape * loss_shape + \
                   weight_loss_hand * loss_hand + \
                   weight_collision * loss_collision + weight_loss_contact * loss_contact
            loss.backward(retain_graph=True)
            optimizer.step()

        print('[INFO] optimization stage 2 finished.')

        #################### visualization body mesh after adv optimization (stafe 2) ################
        # smplx params --> body mesh
        body_params_opt_s2 = convert_to_3D_rot(body_params_rec)  # tensor, [bs=1, 72]
        body_pose_joint = vposer_model.decode(body_params_opt_s2[:, 16:48], output_type='aa').view(1, -1)
        body_verts_opt_s2 = gen_body_mesh(body_params_opt_s2, body_pose_joint, smplx_model)[0]
        body_verts_opt_s2 = body_verts_opt_s2.detach().cpu().numpy()   # [n_body_vert, 3]

        body_verts_opt_prox_s2 = np.zeros(body_verts_opt_s2.shape)  # [10475, 3]
        temp = body_verts_opt_s2 - shift
        body_verts_opt_prox_s2[:, 0] = temp[:, 0] * math.cos(-rot_angle_1) - \
                                       temp[:, 1] * math.sin(-rot_angle_1)
        body_verts_opt_prox_s2[:, 1] = temp[:, 0] * math.sin(-rot_angle_1) + \
                                       temp[:, 1] * math.cos(-rot_angle_1)
        body_verts_opt_prox_s2[:, 2] = temp[:, 2]

        body_mesh_opt_s2 = o3d.geometry.TriangleMesh()
        body_mesh_opt_s2.vertices = o3d.utility.Vector3dVector(body_verts_opt_prox_s2)
        body_mesh_opt_s2.triangles = o3d.utility.Vector3iVector(smplx_model.faces)
        body_mesh_opt_s2.compute_vertex_normals()

        o3d.visualization.draw_geometries([scene_mesh, body_mesh_opt_s2])  # use normal open3d visualization

        # visualizer = JVisualizer()  # use webGL
        # visualizer.add_geometry(scene_mesh)
        # visualizer.add_geometry(body_mesh_opt_s2)
        # visualizer.show()





if __name__ == '__main__':
    optimize()