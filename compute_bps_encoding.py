import argparse
import torch
from torch.utils import data
from tqdm import tqdm
from preprocess.preprocess_train import *
from loader.preprocess_loader import PreprocessLoader
from utils import *
import smplx



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=20, help='input batch size')
parser.add_argument("--num_bps", default=10000, type=int, help='# of basis points')
parser.add_argument("--cube_size", default=2.0, type=float, help='size of 3D cage')
parser.add_argument('--smplx_model_path', type=str,
                    default='/mnt/hdd/PROX/body_models/smplx_model',
                    help='path to smplx body model')
parser.add_argument('--dataset_path', type=str,
                    default='/mnt/hdd/PROX',
                    help='path to prox dataset')
parser.add_argument('--preprocess_file_path', type=str,
                    default='/mnt/hdd/PROX/preprocessed_encoding',
                    help='path to preprocessed bps features')
parser.add_argument('--scene_name', type=str, default='BasementSittingBooth', help='current preprocessed scene name')
parser.add_argument("--split", default='train', type=str, choices=['train', 'test'])
parser.add_argument("--auge_per_sample", default=4, type=int, help='# of augmentations for each sample')
# parser.add_argument("--id", default=1, type=int, help='id of different augmentations of the same scene')
args = parser.parse_args()


# test scene: ['MPH1Library', 'MPH16', 'N0SittingBooth', 'N3OpenArea']
# train scene: ['BasementSittingBooth', 'MPH8', 'MPH11', 'MPH112', 'N0Sofa', 'N3Library', 'Werkraum', 'N3Office']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    proxd_path = os.path.join(args.dataset_path, 'PROXD')
    cam2world_path = os.path.join(args.dataset_path, 'cam2world')
    scene_mesh_path = os.path.join(args.dataset_path, 'scenes_downsampled')

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
                               batch_size=args.batch_size
                               ).to(device)
    print('[INFO] smplx model loaded.')

    if not os.path.exists(args.preprocess_file_path):
        os.makedirs(args.preprocess_file_path)


    ##################### get body verts for all samples #################
    dataset = PreprocessLoader(scene_name=args.scene_name)
    dataset.load_body_params(proxd_path)
    dataset.load_cam_params(cam2world_path)
    dataset.load_scene(scene_mesh_file=scene_mesh_path)
    bps_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=4,
                                                 drop_last=True)

    body_verts_list = []
    with torch.no_grad():
        for step, data in tqdm(enumerate(bps_dataloader)):
            [body_params, body_pose_joint, cam_pose, scene_verts] = [item.to(device) for item in data]
            body_verts = gen_body_mesh(body_params, body_pose_joint, smplx_model)  # [bs, n_body_verts, 3]
            body_verts = verts_transform(body_verts, cam_pose)
            # save valid data (if the body vertex is within the scene)
            for i in range(body_verts.shape[0]):
                if np.max(np.abs(body_verts[i].cpu().numpy())) <= np.max(np.abs(scene_verts[i].cpu().numpy())):
                    body_verts_list.append(body_verts[i].cpu().numpy())  # each element in list: [1, n_verts, 3]
    print('[INFO] body mesh ready.')


    ################### calculate scene/body bps representation ########
    n_sample = len(body_verts_list)
    cur_scene = o3d.io.read_triangle_mesh(os.path.join(scene_mesh_path, args.scene_name + '.ply'))
    cur_scene_verts = np.asarray(cur_scene.vertices)

    ############# crop scene around human body ##########
    scene_verts_crop_local_list, scene_verts_local_list, body_verts_local_list, = [], [], []
    shift_list_1, rot_angle_list_1 = [], []

    for itr in range(args.auge_per_sample):
        for i in tqdm(range(n_sample)):
            # rotate scene / body
            scene_verts, body_verts, rot_angle = rotate_scene_smplx(cur_scene_verts, body_verts_list[i])
            # crop scene, shift verts
            scene_verts_local, scene_verts_crop_local, _, body_verts_local, shift = \
                crop_scene_cube_smplx(scene_verts, body_verts, r=args.cube_size)

            scene_verts_crop_local_list.append(scene_verts_crop_local)
            scene_verts_local_list.append(scene_verts_local)  # list, [n_sample*4, n_scene_verts, 3]
            body_verts_local_list.append(body_verts_local)  # list, [n_sample*4, n_body_verts, 3]

            rot_angle_list_1.append(rot_angle)
            shift_list_1.append(shift)

    print('[INFO] scene mesh cropped and shifted.')
    del body_verts_list

    ######################## calculate bps ##############
    scene_bps_list, body_bps_list = [], []
    scene_bps_verts_local_list = []
    scene_basis_set = bps_gen_ball_inside(n_bps=args.num_bps, random_seed=100)
    for i in tqdm(range(len(body_verts_local_list))):
        scene_verts_global, scene_verts_crop_global, body_verts_global, _ = \
            augmentation_crop_scene_smplx(scene_verts_local_list[i] / args.cube_size,
                                          scene_verts_crop_local_list[i] / args.cube_size,
                                          body_verts_local_list[i] / args.cube_size,
                                          split=args.split,
                                          scale=args.cube_size)

        scene_bps, selected_scene_verts_global, selected_ind = bps_encode_scene(scene_basis_set, scene_verts_crop_global)
        body_bps = bps_encode_body(selected_scene_verts_global, body_verts_global)  # [n_feat, n_bps]
        scene_bps_list.append(scene_bps)
        body_bps_list.append(body_bps)

        selected_scene_verts_local = scene_verts_crop_local_list[i][selected_ind]
        scene_bps_verts_local_list.append(selected_scene_verts_local)

    print('[INFO] scene/body bps representation ready (with augmentation).')


    ########################### save preprocessed data ##############################
    # if run compute_bps_encoding.py multiple times for each scene
    # use different prefix for saving each time (ex. 'BasementSittingBooth_1_...', 'BasementSittingBooth_2_...')
    rot_angle_list_1 = np.asarray(rot_angle_list_1)
    shift_list_1 = np.asarray(shift_list_1)
    np.save('{}/{}_rot_list.npy'.format(args.preprocess_file_path, args.scene_name), rot_angle_list_1)
    np.save('{}/{}_shift_list.npy'.format(args.preprocess_file_path, args.scene_name), shift_list_1)
    del rot_angle_list_1
    del shift_list_1

    scene_bps_list = np.asarray(scene_bps_list)  # [n_sample*4, n_feat, n_bps]
    np.save('{}/{}_scene_bps_list.npy'.format(args.preprocess_file_path, args.scene_name), scene_bps_list)
    del scene_bps_list

    body_bps_list = np.asarray(body_bps_list)
    np.save('{}/{}_body_bps_list.npy'.format(args.preprocess_file_path, args.scene_name), body_bps_list)
    del body_bps_list

    body_verts_local_list = np.asarray(body_verts_local_list)
    np.save('{}/{}_body_verts_local_list.npy'.format(args.preprocess_file_path, args.scene_name), body_verts_local_list)
    del body_verts_local_list

    scene_bps_verts_local_list = np.asarray(scene_bps_verts_local_list)
    np.save('{}/{}_scene_bps_verts_local_list.npy'.format(args.preprocess_file_path, args.scene_name), scene_bps_verts_local_list)
    del scene_bps_verts_local_list


    print('[INFO] preprocessed bps/verts saved to {} for scene {}'.format(args.preprocess_file_path, args.scene_name))




