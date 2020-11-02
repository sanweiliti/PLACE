import numpy as np
import torch
from torch.utils import data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestLoader(data.Dataset):
    def __init__(self):
        self.scene_bps_list = []
        self.scene_bps_verts_list = []
        self.n_samples = 0


    def __len__(self):
        return self.n_samples


    def __getitem__(self, index):
        scene_bps = torch.from_numpy(self.scene_bps_list[index]).float()  # [n_feat, n_bps]
        scene_bps_verts = torch.from_numpy(self.scene_bps_verts_list[index].transpose(1, 0)).float()
        return [scene_bps, scene_bps_verts]

