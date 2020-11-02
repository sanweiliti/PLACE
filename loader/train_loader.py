import numpy as np
import torch
from torch.utils import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainLoader(data.Dataset):
    def __init__(self, mode='body_bps_verts'):
        self.scene_bps_list = []
        self.body_bps_list = []
        self.body_verts_list = []
        self.scene_bps_verts_list = []
        self.body_params_list = []
        self.shift_list = []
        self.rotate_list = []
        self.scene_name_list = []
        self.grid_min_dict = dict()
        self.grid_max_dict = dict()
        self.sdf_dict = dict()

        # self.scene_verts_list = []
        self.scene_verts_dict = dict()

        self.n_samples = 0
        self.mode = mode


    def __len__(self):
        return self.n_samples


    def __getitem__(self, index):
        scene_bps = torch.from_numpy(self.scene_bps_list[index]).float()  # [n_feat, n_bps]
        body_bps = torch.from_numpy(self.body_bps_list[index]).float()  # [n_feat, n_bps]

        if self.mode == 'body_verts':
            body_verts = torch.from_numpy(self.body_verts_list[index].transpose(1, 0)).float()  # [3, n_body_verts]
            scene_bps_verts = torch.from_numpy(self.scene_bps_verts_list[index].transpose(1, 0)).float()
            return [body_verts, scene_bps, body_bps, scene_bps_verts]

        elif self.mode == 'body_verts_contact':
            body_verts = torch.from_numpy(self.body_verts_list[index].transpose(1, 0)).float()  # [3, n_body_verts]
            scene_bps_verts = torch.from_numpy(self.scene_bps_verts_list[index].transpose(1, 0)).float()
            shift = torch.from_numpy(self.shift_list[index]).float()  # tensor, [3]
            rotate = torch.from_numpy(np.array([self.rotate_list[index]])).float()  # tensor, [1]
            scene_verts = torch.from_numpy(self.scene_verts_dict[self.scene_name_list[index]]).float()  # [50000, 3]
            return [body_verts, scene_bps, body_bps, scene_bps_verts, shift, rotate, scene_verts]

        elif self.mode == 'body_params_contact':
            body_params = torch.from_numpy(self.body_params_list[index]).float()  # [72]
            scene_bps_verts = torch.from_numpy(self.scene_bps_verts_list[index].transpose(1, 0)).float()
            shift = torch.from_numpy(self.shift_list[index]).float()  # tensor, [3]
            rotate = torch.from_numpy(np.array([self.rotate_list[index]])).float()  # tensor, [1]
            scene_verts = torch.from_numpy(self.scene_verts_dict[self.scene_name_list[index]]).float()  # [50000, 3]
            return [body_params, scene_bps, body_bps, scene_bps_verts, shift, rotate, scene_verts]

        elif self.mode == 'body_verts_contact_coll':
            body_verts = torch.from_numpy(self.body_verts_list[index].transpose(1, 0)).float()  # [3, n_body_verts]
            scene_bps_verts = torch.from_numpy(self.scene_bps_verts_list[index].transpose(1, 0)).float()
            shift = torch.from_numpy(self.shift_list[index]).float()  # tensor, [3]
            rotate = torch.from_numpy(np.array([self.rotate_list[index]])).float()  # tensor, [1]
            scene_name = self.scene_name_list[index]
            scene_grid_min = self.grid_min_dict[scene_name]
            scene_grid_max = self.grid_max_dict[scene_name]
            scene_grid_sdf = self.sdf_dict[scene_name]

            scene_verts = torch.from_numpy(self.scene_verts_dict[self.scene_name_list[index]]).float()  # [50000, 3]
            return [body_verts, scene_bps, body_bps, scene_bps_verts, shift, rotate, scene_verts,
                    scene_grid_min, scene_grid_max, scene_grid_sdf]
