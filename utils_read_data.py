import os
import open3d as o3d
import numpy as np
import json
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# use for test/optimize
def read_mesh_sdf(dataset_path, dataset, scene_name):
    if dataset == 'prox':
        scene_mesh_path = os.path.join(dataset_path, 'scenes_downsampled')
        scene = o3d.io.read_triangle_mesh(os.path.join(scene_mesh_path, scene_name + '.ply'))
        cur_scene_verts = np.asarray(scene.vertices)

        ## read scene sdf
        scene_sdf_path = os.path.join(dataset_path, 'sdf')
        with open(os.path.join(scene_sdf_path, scene_name + '.json')) as f:
            sdf_data = json.load(f)
            grid_min = np.array(sdf_data['min'])
            grid_max = np.array(sdf_data['max'])
            grid_dim = sdf_data['dim']
        sdf = np.load(os.path.join(scene_sdf_path, scene_name + '_sdf.npy')).reshape(grid_dim, grid_dim, grid_dim)
        s_grid_min_batch = torch.tensor(grid_min, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        s_grid_max_batch = torch.tensor(grid_max, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        s_sdf_batch = torch.tensor(sdf, dtype=torch.float32, device=device).unsqueeze(0)
        s_sdf_batch = s_sdf_batch.repeat(1, 1, 1, 1)  # [1, 256, 256, 256]


    elif dataset == 'mp3d':
        scene = o3d.io.read_triangle_mesh(os.path.join(dataset_path, scene_name + '.ply'))
        # swap z, y axis
        cur_scene_verts = np.zeros(np.asarray(scene.vertices).shape)
        cur_scene_verts[:, 0] = np.asarray(scene.vertices)[:, 0]
        cur_scene_verts[:, 1] = np.asarray(scene.vertices)[:, 2]
        cur_scene_verts[:, 2] = np.asarray(scene.vertices)[:, 1]

        ## read scene sdf
        scene_sdf_path = os.path.join(dataset_path, 'sdf')
        with open(os.path.join(scene_sdf_path, scene_name + '.json')) as f:
            sdf_data = json.load(f)
            grid_min = np.array(sdf_data['min'])
            grid_max = np.array(sdf_data['max'])
            grid_min = np.array([grid_min[0], grid_min[2], grid_min[1]])
            grid_max = np.array([grid_max[0], grid_max[2], grid_max[1]])
            grid_dim = sdf_data['dim']
        sdf = np.load(os.path.join(scene_sdf_path, scene_name + '_sdf.npy')).reshape(grid_dim, grid_dim, grid_dim)
        sdf = sdf.transpose(0, 2, 1)  # swap y,z axis
        s_grid_min_batch = torch.tensor(grid_min, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        s_grid_max_batch = torch.tensor(grid_max, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        s_sdf_batch = torch.tensor(sdf, dtype=torch.float32, device=device).unsqueeze(0)
        s_sdf_batch = s_sdf_batch.repeat(1, 1, 1, 1)


    elif dataset == 'replica':
        scene = o3d.io.read_triangle_mesh(os.path.join(os.path.join(dataset_path, scene_name), 'mesh.ply'))
        cur_scene_verts = np.asarray(scene.vertices)

        ## read scene sdf
        scene_sdf_path = os.path.join(dataset_path, 'sdf')
        with open(os.path.join(scene_sdf_path, scene_name + '.json')) as f:
            sdf_data = json.load(f)
            grid_min = np.array(sdf_data['min'])
            grid_max = np.array(sdf_data['max'])
            grid_dim = sdf_data['dim']
        sdf = np.load(os.path.join(scene_sdf_path, scene_name + '_sdf.npy')).reshape(grid_dim, grid_dim, grid_dim)
        s_grid_min_batch = torch.tensor(grid_min, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        s_grid_max_batch = torch.tensor(grid_max, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        s_sdf_batch = torch.tensor(sdf, dtype=torch.float32, device=device).unsqueeze(0)
        s_sdf_batch = s_sdf_batch.repeat(1, 1, 1, 1)

    return scene, cur_scene_verts, s_grid_min_batch, s_grid_max_batch, s_sdf_batch



# rotate scene to be parallel with x,y axis, and define scene boundary
def define_scene_boundary(dataset, scene_name):
    if dataset == 'prox':
        if scene_name == 'MPH1Library':
            rot_angle = 0.4
            scene_min_x = -1.3
            scene_max_x = 2.8
            scene_min_y = -1.8
            scene_max_y = 1.2
        elif scene_name == 'MPH16':
            rot_angle = 0.5
            scene_min_x = -1.5
            scene_max_x = 1.3
            scene_min_y = -1.7
            scene_max_y = 2.4
        elif scene_name == 'N0SittingBooth':
            rot_angle = -0.25
            scene_min_x = -0.4
            scene_max_x = 4.1
            scene_min_y = -1.6
            scene_max_y = 0.7
        elif scene_name == 'N3OpenArea':
            rot_angle = -0.1
            scene_min_x = -1.4
            scene_max_x = 2.1
            scene_min_y = -2.4
            scene_max_y = 1.2

    elif dataset == 'mp3d':
        if scene_name == '17DRP5sb8fy-bedroom':
            rot_angle = 0
            scene_min_x = -11.1
            scene_max_x = -6.5
            scene_min_y = -5.1
            scene_max_y = -0.2
        elif scene_name == '17DRP5sb8fy-familyroomlounge':
            rot_angle = 0
            scene_min_x = -11.4
            scene_max_x = -8.0
            scene_min_y = -0.8
            scene_max_y = 2.8
        elif scene_name == '17DRP5sb8fy-livingroom':
            rot_angle = 0
            scene_min_x = -6.8
            scene_max_x = -1.8
            scene_min_y = -5.3
            scene_max_y = -0.5
        elif scene_name == 'sKLMLpTHeUy-familyname_0_1':
            rot_angle = 0
            scene_min_x = 0.0
            scene_max_x = 7.9
            scene_min_y = -4.5
            scene_max_y = 2.4
        elif scene_name == 'X7HyMhZNoso-livingroom_0_16':
            rot_angle = 0
            scene_min_x = -6.2
            scene_max_x = -1.4
            scene_min_y = 11.2
            scene_max_y = 16.7
        elif scene_name == 'zsNo4HB9uLZ-bedroom0_0':
            rot_angle = 0
            scene_min_x = -2.3
            scene_max_x = 1.6
            scene_min_y = -7.9
            scene_max_y = -4.0
        elif scene_name == 'zsNo4HB9uLZ-livingroom0_13':
            rot_angle = 0
            scene_min_x = 4.0
            scene_max_x = 8.5
            scene_min_y = 0.6
            scene_max_y = 4.2

    elif dataset == 'replica':
        if scene_name == 'office_2':
            rot_angle = 0.42
            scene_min_x = -2.5
            scene_max_x = 1.0
            scene_min_y = -2.3
            scene_max_y = 4.5
        elif scene_name == 'hotel_0':
            rot_angle = 0.05
            scene_min_x = -2.8
            scene_max_x = 5.2
            scene_min_y = -1.5
            scene_max_y = 1.9
        elif scene_name == 'room_0':
            rot_angle = 0
            scene_min_x = -1.0
            scene_max_x = 6.8
            scene_min_y = -1.2
            scene_max_y = 3.3
        elif scene_name == 'frl_apartment_0':
            rot_angle = 0
            scene_min_x = -1.0
            scene_max_x = 6.0
            scene_min_y = -5.2
            scene_max_y = 3.0
        elif scene_name == 'apartment_1':
            rot_angle = 0
            scene_min_x = -2.5
            scene_max_x = 8.0
            scene_min_y = -1.0
            scene_max_y = 6.7
    
    return rot_angle, scene_min_x, scene_max_x, scene_min_y, scene_max_y
