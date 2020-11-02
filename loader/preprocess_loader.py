import open3d as o3d
import json
import os, glob
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class PreprocessLoader(data.Dataset):
    def __init__(self,
                 # split='train',    # train/test/
                 scene_name='BasementSittingBooth',
                 with_bps=False,
                 ):
        self.scene_name = scene_name
        print('[INFO] {} scene selected'.format(self.scene_name))
        self.with_bps = with_bps
        if self.with_bps:
            self.scene_bps = []
            self.body_bps = []
            self.body_verts = []
            self.scene_verts = []


    def load_body_params(self, proxd_path):
        self.body_params_list = []
        self.body_pose_joint_list = []

        # file names in PROXD: ex. MPH112_00034_01, 54 in total
        body_file_list = glob.glob(os.path.join(proxd_path, '*'))
        body_file_list.sort()
        for bodyfile_perscene in tqdm(body_file_list):  # bodyfile_perscene: ex. ../MPH112_00034_01
            scene_name = bodyfile_perscene.split("/")[-1][:-9]
            if scene_name == self.scene_name:
                bodyfile_perframe = os.listdir(os.path.join(bodyfile_perscene, 'results'))
                bodyfile_perframe.sort()
                for file_path in bodyfile_perframe:  # file_path: ex. s001_frame_00914__00.00.30.436
                    file_path = os.path.join(bodyfile_perscene, 'results', file_path, '000.pkl')
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        body_params = np.concatenate((data['transl'][0], data['global_orient'][0], data['betas'][0],
                                                      data['pose_embedding'][0],
                                                      data['left_hand_pose'][0], data['right_hand_pose'][0]), axis=0)  # [72,]
                        body_pose_joint = data['body_pose'][0]  # [63,]
                        if np.sum(np.isnan(body_params)) == 0 and np.sum(np.isnan(body_pose_joint)) == 0:
                            self.body_params_list.append(body_params)           # [n_samples]
                            self.body_pose_joint_list.append(body_pose_joint)   # [n_samples]

        # a=int(0.99*len(self.body_params_list))
        # self.body_params_list = self.body_params_list[a:]
        # self.body_pose_joint_list = self.body_pose_joint_list[a:]

        self.n_samples = len(self.body_params_list)
        print('[INFO] body params loaded, read n_samples={:d}'.format(self.n_samples))


    def load_cam_params(self, cam2world_file):
        with open(os.path.join(cam2world_file, self.scene_name + '.json'), 'r') as f:
            cam_pose = np.array(json.load(f))
            self.cam_pose = cam_pose
        print('[INFO] camera pose loaded.')


    def load_scene(self, scene_mesh_file):
        scene_o3d = o3d.io.read_triangle_mesh(os.path.join(scene_mesh_file, self.scene_name + '.ply'))
        self.scene = {}
        self.scene['verts'] = np.asarray(scene_o3d.vertices)
        self.scene['cam_pose'] = self.cam_pose
        print('[INFO] scene mesh files loaded.')


    def __len__(self):
        return self.n_samples


    def __getitem__(self, index):
        body_params = torch.from_numpy(self.body_params_list[index]).float()  # [72]
        body_pose_joint = torch.from_numpy(self.body_pose_joint_list[index]).float()  # [63]
        scene_verts = torch.from_numpy(self.scene['verts']).float()   # [50000, 3]
        cam_pose = torch.from_numpy(self.scene['cam_pose']).float()        # [4, 4]

        if not self.with_bps:
            return [body_params, body_pose_joint, cam_pose, scene_verts]
        else:
            scene_verts = torch.from_numpy(self.scene_verts[index]).float()
            body_verts = torch.from_numpy(self.body_verts[index]).float()  # [3, n_body_verts], scaled
            scene_bps = torch.from_numpy(self.scene_bps[index]).float()  # [n_feat, n_bps]
            body_bps = torch.from_numpy(self.body_bps[index]).float()  # [n_feat, n_bps]
            return [body_params, body_pose_joint, cam_pose, scene_verts, body_verts, scene_bps, body_bps]





if __name__ == '__main__':
    dataset_path = '/mnt/hdd/Siwei/proxe'
    scene_mesh_path = os.path.join(dataset_path, 'scenes_downsampled')
    scene_sdf_path = os.path.join(dataset_path, 'scenes_sdf')
    proxd_path = '/mnt/hdd/PROX/PROXD'
    cam2world_path = os.path.join('/mnt/hdd/PROX/cam2world')

    dataset = PreprocessLoader()
    dataset.load_body_params(proxd_path)
    dataset.load_cam_params(cam2world_path)
    dataset.load_scene(scene_mesh_file=scene_mesh_path)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=True, num_workers=2, drop_last=True)

    num = 0
    for epoch in range(5):
        for data in dataloader:
            body_params, body_pose_joint, cam_pose, scene_verts, \
            scene_faces, scene_grid_min, scene_grid_max, scene_grid_sdf = [item.to(device) for item in data[:-1]]



