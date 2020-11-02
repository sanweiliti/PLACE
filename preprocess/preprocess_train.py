import numpy as np
import smplx
from human_body_prior.tools.model_loader import load_vposer
import torch
from loader.preprocess_loader import PreprocessLoader
import random
from tqdm import tqdm
import open3d as o3d
from torch.utils import data
import math
from utils import *
from preprocess.bps_encoding import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# rotate before cropping scene cube
def rotate_scene_smplx(scene_verts, body_verts):
    # random rotate around z-axis
    rot_angle = random.uniform(0, 2 * (math.pi))
    scene_verts_aug = np.zeros(scene_verts.shape)
    body_verts_aug = np.zeros(body_verts.shape)

    scene_verts_aug[:, 0] = scene_verts[:, 0] * math.cos(rot_angle) - scene_verts[:, 1] * math.sin(rot_angle)
    scene_verts_aug[:, 1] = scene_verts[:, 0] * math.sin(rot_angle) + scene_verts[:, 1] * math.cos(rot_angle)
    scene_verts_aug[:, 2] = scene_verts[:, 2]
    body_verts_aug[:, 0] = body_verts[:, 0] * math.cos(rot_angle) - body_verts[:, 1] * math.sin(rot_angle)
    body_verts_aug[:, 1] = body_verts[:, 0] * math.sin(rot_angle) + body_verts[:, 1] * math.cos(rot_angle)
    body_verts_aug[:, 2] = body_verts[:, 2]

    return scene_verts_aug, body_verts_aug, rot_angle



def crop_scene_cube_smplx(scene_verts, body_verts, r=2.0, with_wall_ceilling=True):
    # select body center as initial cube center in x,y axis
    body_center = np.mean(body_verts, axis=0)
    # random shift the cube center in x,y axis
    random_shift = np.array([0.0, 0.0, 0.0])
    body_min_x = np.min(body_verts[:, 0])
    body_max_x = np.max(body_verts[:, 0])
    body_min_y = np.min(body_verts[:, 1])
    body_max_y = np.max(body_verts[:, 1])
    random_shift[0] = random.uniform(max(-r/3, (body_max_x - body_center[0]) - r/2),
                                     min(r/3, r/2 - (body_center[0] - body_min_x)))
    random_shift[1] = random.uniform(max(-r/3, (body_max_y - body_center[1]) - r/2),
                                     min(r/3, r/2 - (body_center[1] - body_min_y)))

    min_x = body_center[0] - r/2 + random_shift[0]
    max_x = body_center[0] + r/2 + random_shift[0]
    min_y = body_center[1] - r/2 + random_shift[1]
    max_y = body_center[1] + r/2 + random_shift[1]

    # cropped scene verts
    scene_verts_crop = scene_verts[np.where((scene_verts[:,0] >= min_x) & (scene_verts[:,0] <= max_x) &
                                            (scene_verts[:,1] >= min_y) & (scene_verts[:,1] <= max_y))]

    scene_center = body_center + random_shift  # define cube center in x,y axis
    scene_center[2] = np.min(scene_verts[:, 2]) + 1.0  # fix cube center 1m above ground in z axis

    scene_verts_crop = scene_verts_crop - scene_center
    scene_verts = scene_verts - scene_center
    body_verts = body_verts - scene_center

    scene_verts_crop_nowall = scene_verts_crop
    # add ceiling/walls to scene_verts_crop, define scene cage
    if with_wall_ceilling:
        n_pts_edge = 70
        grid = (max_x - min_x) / n_pts_edge
        ceiling_points, wall1_points, wall2_points, wall3_points, wall4_points = [], [], [], [], []
        for i in range(n_pts_edge):
            for j in range(n_pts_edge):
                x = min_x + (i + 1) * grid - scene_center[0]
                y = min_y + (j + 1) * grid - scene_center[1]
                ceiling_points.append(np.array([x, y, 1.0]))  # ceiling hight: 1m above cube center(origin)
        for i in range(n_pts_edge):
            for j in range(n_pts_edge):
                x = min_x + (i + 1) * grid - scene_center[0]
                z = -1.0 + (j + 1) * grid
                wall1_points.append(np.array([x, min_y - scene_center[1], z]))
        for i in range(n_pts_edge):
            for j in range(n_pts_edge):
                x = min_x + (i + 1) * grid - scene_center[0]
                z = -1.0 + (j + 1) * grid
                wall2_points.append(np.array([x, max_y - scene_center[1], z]))
        for i in range(n_pts_edge):
            for j in range(n_pts_edge):
                y = min_y + (i + 1) * grid - scene_center[1]
                z = -1.0 + (j + 1) * grid
                wall3_points.append(np.array([min_x - scene_center[0], y, z]))
        for i in range(n_pts_edge):
            for j in range(n_pts_edge):
                y = min_y + (i + 1) * grid - scene_center[1]
                z = -1.0 + (j + 1) * grid
                wall4_points.append(np.array([max_x - scene_center[0], y, z]))
        ceiling_points = np.asarray(ceiling_points)  # [n_ceiling_pts, 3]
        wall1_points = np.asarray(wall1_points)
        wall2_points = np.asarray(wall2_points)
        wall3_points = np.asarray(wall3_points)
        wall4_points = np.asarray(wall4_points)

        scene_verts_crop = np.concatenate((scene_verts_crop, ceiling_points, wall1_points, wall2_points, wall3_points, wall4_points), axis=0)

    shift = -scene_center
    return scene_verts, scene_verts_crop, scene_verts_crop_nowall, body_verts, shift



def scale_cropped_one_scene_smplx(scene_verts_list, scene_verts_crop_list, body_verts_list, scale):
    max_dist = scale
    scaled_scene_verts_crop_list = [scene_verts / max_dist for scene_verts in scene_verts_crop_list]
    scaled_scene_verts_list = [scene_verts / max_dist for scene_verts in scene_verts_list]
    scaled_body_verts_list = [body_verts / max_dist for body_verts in body_verts_list]
    return scaled_scene_verts_list, scaled_scene_verts_crop_list, scaled_body_verts_list



def augmentation_crop_scene_smplx(scene_verts, scene_verts_crop, body_verts, split='train', scale=2.0):
    # random rotate around z-axis
    rot_angle = random.uniform(0, 2*(math.pi))
    scene_verts_global = np.zeros(scene_verts.shape)
    scene_verts_crop_global = np.zeros(scene_verts_crop.shape)
    body_verts_global = np.zeros(body_verts.shape)

    scene_verts_global[:, 0] = scene_verts[:, 0] * math.cos(rot_angle) - scene_verts[:, 1] * math.sin(rot_angle)
    scene_verts_global[:, 1] = scene_verts[:, 0] * math.sin(rot_angle) + scene_verts[:, 1] * math.cos(rot_angle)
    scene_verts_global[:, 2] = scene_verts[:, 2]
    scene_verts_crop_global[:, 0] = scene_verts_crop[:, 0] * math.cos(rot_angle) - scene_verts_crop[:, 1] * math.sin(rot_angle)
    scene_verts_crop_global[:, 1] = scene_verts_crop[:, 0] * math.sin(rot_angle) + scene_verts_crop[:, 1] * math.cos(rot_angle)
    scene_verts_crop_global[:, 2] = scene_verts_crop[:, 2]
    body_verts_global[:, 0] = body_verts[:, 0] * math.cos(rot_angle) - body_verts[:, 1] * math.sin(rot_angle)
    body_verts_global[:, 1] = body_verts[:, 0] * math.sin(rot_angle) + body_verts[:, 1] * math.cos(rot_angle)
    body_verts_global[:, 2] = body_verts[:, 2]

    # random small shift for x, y, z
    if split == 'train':
        shift = np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)])
        scene_verts_global = scene_verts_global + shift
        scene_verts_crop_global = scene_verts_crop_global + shift
        body_verts_global = body_verts_global + shift

    return scene_verts_global, scene_verts_crop_global, body_verts_global, rot_angle







############################### for debug & visualization ####################################
if __name__ == '__main__':
    batch_size = 1
    n_bps = 10000

    dataset_path = '/Users/siwei/Desktop/3DHuman_gen/dataset/proxe'
    proxd_path = os.path.join(dataset_path, 'PROXD')
    cam2world_path = os.path.join(dataset_path, 'cam2world')
    vposer_model_path = os.path.join(dataset_path, 'body_models/vposer_v1_0')
    smplx_model_path = os.path.join(dataset_path, 'body_models/smplx_model')

    scene_mesh_path = os.path.join(dataset_path, 'scenes_downsampled')
    scene_sdf_path = os.path.join(dataset_path, 'scenes_sdf')

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
                               batch_size=batch_size
                               ).to(device)
    print('[INFO] smplx model loaded.')

    vposer_model, _ = load_vposer(vposer_model_path, vp_model='snapshot')
    vposer_model = vposer_model.to(device)
    print('[INFO] vposer model loaded')


    ######## set body mesh gen dataloader ###########
    scene_name = 'MPH1Library'
    dataset = PreprocessLoader(scene_name=scene_name)
    dataset.load_body_params(proxd_path)
    dataset.load_cam_params(cam2world_path)
    dataset.load_scene(scene_mesh_file=scene_mesh_path)
    bps_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                                 drop_last=True)  # drop_last=T:  smlpx model needs to predefine bs

    ######### get body mesh for all samples ###########
    body_verts_list = []
    cam_pose_list = []
    with torch.no_grad():
        for step, data in tqdm(enumerate(bps_dataloader)):
            [body_params, body_pose_joint, cam_pose, scene_verts] = [item.to(device) for item in data]
            body_verts = gen_body_mesh(body_params, body_pose_joint, smplx_model)  # [bs, n_body_verts, 3]
            body_verts = verts_transform(body_verts, cam_pose)
            # save valid data (if the generated body vertex is within the scene)
            for i in range(body_verts.shape[0]):
                if np.max(np.abs(body_verts[i].cpu().numpy())) <= np.max(np.abs(scene_verts[i].cpu().numpy())):
                    body_verts_list.append(body_verts[i].cpu().numpy())  # each element in list: [1, n_verts, 3]
    print('[INFO] body mesh ready.')


    ########## calculate scene/body bps representation ############
    n_sample = len(body_verts_list)

    cur_scene = o3d.io.read_triangle_mesh(os.path.join(scene_mesh_path, scene_name + '.ply'))
    cur_scene_verts = np.asarray(cur_scene.vertices)
    cur_scene_faces = np.asarray(cur_scene.triangles)

    cube_size = 2.0

    # crop scene around each human body
    scene_verts_crop_local_list, scene_verts_local_list, body_verts_local_list, = [], [], []
    for i in tqdm(range(n_sample)):
        # rotate scene / body
        scene_verts, body_verts, rot_angle = rotate_scene_smplx(cur_scene_verts, body_verts_list[i])
        # crop scene, shift verts
        scene_verts_local, scene_verts_crop_local, _, body_verts_local, shift = \
            crop_scene_cube_smplx(scene_verts, body_verts, r=cube_size)

        scene_verts_crop_local_list.append(scene_verts_crop_local)  # list, different number of verts for each cropped scene
        scene_verts_local_list.append(scene_verts_local)  # list, [[n_sample, n_scene_verts, 3]]
        body_verts_local_list.append(body_verts_local)  # list, [n_sample, n_body_verts, 3]

    print('[INFO] scene mesh cropped and shifted.')



    ######################## calculate bps ##############
    scene_bps_list, body_bps_list = [], []
    scene_verts_global_list, scene_verts_crop_global_list, body_verts_global_list = [], [], []
    scene_bps_verts_global_list, scene_bps_verts_local_list = [], []
    scene_basis_set = bps_gen_ball_inside(n_bps=n_bps, random_seed=100)
    # encode bps
    for i in tqdm(range(n_sample)):
        scene_verts_global, scene_verts_crop_global, body_verts_global, _ = \
            augmentation_crop_scene_smplx(scene_verts_local_list[i] / cube_size,
                                          scene_verts_crop_local_list[i] / cube_size,
                                          body_verts_local_list[i] / cube_size,
                                          split='train', scale=cube_size)

        scene_verts_global_list.append(scene_verts_global)
        scene_verts_crop_global_list.append(scene_verts_crop_global)  # list
        body_verts_global_list.append(body_verts_global)

        scene_bps, selected_scene_verts_global, selected_ind = bps_encode_scene(scene_basis_set, scene_verts_crop_global)  # [n_feat, n_bps]
        body_bps = bps_encode_body(selected_scene_verts_global, body_verts_global)  # [n_feat, n_bps]
        scene_bps_list.append(scene_bps)
        body_bps_list.append(body_bps)

        selected_scene_verts_local = scene_verts_crop_local_list[i][selected_ind]
        scene_bps_verts_local_list.append(selected_scene_verts_local)
        scene_bps_verts_global_list.append(selected_scene_verts_global)

    scene_bps_list = np.asarray(scene_bps_list)  # [n_sample*4, n_feat, n_bps]
    body_bps_list = np.asarray(body_bps_list)
    scene_bps_verts_local_list = np.asarray(scene_bps_verts_local_list)
    body_verts_local_list = np.asarray(body_verts_local_list)

    scene_verts_global_list = np.asarray(scene_verts_global_list)  # [n_sample, n_verts, 3], verts of whole scene
    body_verts_global_list = np.asarray(body_verts_global_list)  # [n_sample, n_verts, 3]
    scene_bps_verts_global_list = np.asarray(scene_bps_verts_global_list)
    print('[INFO] scene/body bps representation ready (with augmentation).')


    ######### set train dataloader ##############
    dataset.with_bps = True
    dataset.n_samples = n_sample
    dataset.scene_bps = scene_bps_list  # [n_sample, n_feat, n_bps]
    dataset.body_bps = body_bps_list  # [n_sample, n_feat, n_bps]
    dataset.body_verts = body_verts_local_list
    dataset.scene_verts = scene_verts_local_list
    print('[INFO] dataloader updated, select n_samples={}'.format(dataset.__len__()))
    train_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                                                   num_workers=4,drop_last=True)

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
                               batch_size=batch_size
                               ).to(device)

    for step, data in tqdm(enumerate(train_dataloader)):
        [body_params, body_pose_joint, cam_pose, scene_verts, body_verts, _, _] = [item.to(device) for item in data]

        # body_pose_joint = vposer_model.decode(body_params[:,16:48], output_type='aa').view(batch_size, -1)  # [bs, 63]
        # body_verts = gen_body_mesh(body_params, body_pose_joint, smplx_model)[0].detach().cpu().numpy()
        # body_smplx = o3d.geometry.TriangleMesh()
        # triangles = smplx_model.faces
        # body_smplx.vertices = o3d.utility.Vector3dVector(body_verts)
        # body_smplx.triangles = o3d.utility.Vector3iVector(triangles)
        # body_smplx.compute_vertex_normals()


        # visualization for each sample
        body = o3d.geometry.TriangleMesh()
        verts = body_verts_local_list[step*batch_size]
        triangles = smplx_model.faces
        body.vertices = o3d.utility.Vector3dVector(verts)
        body.triangles = o3d.utility.Vector3iVector(triangles)
        body.compute_vertex_normals()
        # body.transform(trans)  # cam2world trans

        scene = o3d.geometry.TriangleMesh()
        verts = scene_verts_local_list[step*batch_size]
        scene.vertices = o3d.utility.Vector3dVector(verts)
        scene.triangles = o3d.utility.Vector3iVector(cur_scene_faces)
        scene.compute_vertex_normals()

        scene_basis_set_pointcloud = o3d.geometry.PointCloud()
        scene_basis_set_pointcloud.points = o3d.utility.Vector3dVector(scene_bps_verts_local_list[step*batch_size])

        # coordinate system
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([body, scene, scene_basis_set_pointcloud, mesh_frame])










