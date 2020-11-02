from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

import torch.nn as nn
import torch.nn.functional as F

# sys.path.append(os.path.join(os.getcwd(), '/torch_mesh_isect/build/lib.linux-x86_64-3.6'))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResBlock(nn.Module):
    def __init__(self, n_channel=1, dim=512):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.fc1 = nn.Linear(in_features=dim, out_features=dim)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.fc2 = nn.Linear(in_features=dim, out_features=dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        out = F.relu(out)
        return out


# AE for scene BPS feature
class BPSRecMLP(nn.Module):
    def __init__(self, n_bps=10000, n_bps_feat=1, hsize1=1024,  hsize2=512):
        super(BPSRecMLP, self).__init__()
        self.n_bps_feat = n_bps_feat    # dimension of BPS feature (always set to 1 in this project)

        # encoder
        self.bn0 = nn.BatchNorm1d(self.n_bps_feat)  # normalize over last axis (n_bps)
        self.fc1 = nn.Linear(in_features=n_bps, out_features=hsize1)
        self.bn1 = nn.BatchNorm1d(self.n_bps_feat)
        self.fc2 = nn.Linear(in_features=hsize1, out_features=hsize2)
        self.bn2 = nn.BatchNorm1d(self.n_bps_feat)
        self.res_block1 = ResBlock(n_channel=self.n_bps_feat, dim=hsize2)
        self.res_block2 = ResBlock(n_channel=self.n_bps_feat, dim=hsize2)

        # decoder
        self.res_block3 = ResBlock(n_channel=self.n_bps_feat, dim=hsize2)
        self.res_block4 = ResBlock(n_channel=self.n_bps_feat, dim=hsize2)
        self.fc3 = nn.Linear(in_features=hsize2, out_features=hsize1)
        self.bn3 = nn.BatchNorm1d(self.n_bps_feat)
        self.fc4 = nn.Linear(in_features=hsize1, out_features=n_bps)
        self.bn4 = nn.BatchNorm1d(self.n_bps_feat)


    def forward(self, x):
        # input x: [bs, n_feat, n_bps]
        x = self.bn0(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.res_block1(x)
        mid_feat = self.res_block2(x)  # middle feature  [bs, n_bps_feat, hsize2]

        dec_feat1 = self.res_block3(mid_feat)               # [bs, n_bps_feat, hsize2]
        dec_feat2 = self.res_block3(dec_feat1)              # [bs, n_bps_feat, hsize2]
        dec_feat3 = F.relu(self.bn3(self.fc3(dec_feat2)))   # [bs, n_bps_feat, hsize1]
        output = F.relu(self.bn4(self.fc4(dec_feat3)))      # [bs, n_bps_feat, n_bps]
        return output, [mid_feat, dec_feat1, dec_feat2, dec_feat3]



# generate body BPS feature
class BPS_CVAE(nn.Module):
    def __init__(self, n_bps=10000, n_bps_feat=1, hsize1=1024,  hsize2=512, eps_d=32):
        super(BPS_CVAE, self).__init__()
        self.n_bps_feat = n_bps_feat
        self.eps_d = eps_d

        # encoder
        self.bn0 = nn.BatchNorm1d(self.n_bps_feat)  # normalize over last axis (n_bps)
        self.fc1 = nn.Linear(in_features=n_bps, out_features=hsize1)
        self.bn1 = nn.BatchNorm1d(self.n_bps_feat)
        self.fc2 = nn.Linear(in_features=hsize1+hsize2, out_features=hsize2)
        self.bn2 = nn.BatchNorm1d(self.n_bps_feat)
        self.res_block1 = ResBlock(n_channel=self.n_bps_feat, dim=hsize2)
        self.res_block2 = ResBlock(n_channel=self.n_bps_feat, dim=hsize2)
        # latent vec
        self.mu_fc = nn.Linear(in_features=hsize2, out_features=self.eps_d)
        self.logvar_fc = nn.Linear(in_features=hsize2, out_features=self.eps_d)

        # decoder
        self.fc3 = nn.Linear(in_features=self.eps_d, out_features=hsize2)
        self.bn3 = nn.BatchNorm1d(self.n_bps_feat)

        self.fc_resblc3 = nn.Linear(in_features=hsize2*2, out_features=hsize2)
        self.bn_resblc3 = nn.BatchNorm1d(self.n_bps_feat)
        self.res_block3 = ResBlock(n_channel=self.n_bps_feat, dim=hsize2)
        self.fc_resblc4 = nn.Linear(in_features=hsize2 * 2, out_features=hsize2)
        self.bn_resblc4 = nn.BatchNorm1d(self.n_bps_feat)
        self.res_block4 = ResBlock(n_channel=self.n_bps_feat, dim=hsize2)

        self.fc4 = nn.Linear(in_features=hsize2*2, out_features=hsize1)
        self.bn4 = nn.BatchNorm1d(self.n_bps_feat)
        self.fc5 = nn.Linear(in_features=hsize1, out_features=n_bps)
        self.bn5 = nn.BatchNorm1d(self.n_bps_feat)

    def _sampler(self, mu, logvar):
        var = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(var.size()).normal_().to(device)
        return eps.mul(var).add_(mu)

    def forward(self, x, scene_feat):
        # input x: [bs, n_feat, n_bps]
        # scene_feat: returned values of BPSRecMLP, [mid_feat, dec_feat1, dec_feat2, dec_feat3]
        x = self.bn0(x)
        x = F.relu(self.bn1(self.fc1(x)))  # [bs, 1, 1024]
        x = F.relu(self.bn2(self.fc2(torch.cat([x, scene_feat[0]], dim=-1))))  # [bs, 1, 512]
        x = F.relu(self.res_block1(x))
        x = F.relu(self.res_block2(x))  # [bs, 1, 512]

        mu = self.mu_fc(x)
        logvar = self.logvar_fc(x)
        x = self._sampler(mu, logvar)  # [bs, 1, eps_d]

        x = F.relu(self.bn3(self.fc3(x)))  # eps_d-->512, [bs, 1, 512]
        x = F.relu(self.bn_resblc3(self.fc_resblc3(torch.cat([x, scene_feat[0]], dim=-1))))  # 1024-->512
        x = self.res_block3(x)    # [bs, 1, 512]
        x = F.relu(self.bn_resblc4(self.fc_resblc4(torch.cat([x, scene_feat[1]], dim=-1))))  # 1024-->512
        x = self.res_block4(x)    # [bs, 1, 512]
        x = F.relu(self.bn4(self.fc4(torch.cat([x, scene_feat[2]], dim=-1))))  # 1024-->1024 [bs, 1, 1024]

        output = F.relu(self.bn5(self.fc5(x)))  # 1024 --> 10000, [bs, 1, 10000]
        return output, mu, logvar


    def sample(self, batch_size, scene_feat):
        eps = torch.randn([batch_size, self.n_bps_feat, self.eps_d], dtype=torch.float32).to(device)
        x = F.relu(self.bn3(self.fc3(eps)))  # eps_d-->512, [bs, 1, 512]
        x = F.relu(self.bn_resblc3(self.fc_resblc3(torch.cat([x, scene_feat[0]], dim=-1))))  # 1024-->512
        x = self.res_block3(x)  # [bs, 1, 512]
        x = F.relu(self.bn_resblc4(self.fc_resblc4(torch.cat([x, scene_feat[1]], dim=-1))))  # 1024-->512
        x = self.res_block4(x)  # [bs, 1, 512]
        x = F.relu(self.bn4(self.fc4(torch.cat([x, scene_feat[2]], dim=-1))))  # 1024-->1024 [bs, 1, 1024]
        sample = F.relu(self.bn5(self.fc5(x)))  # 1024 --> 10000, [bs, 1, 10000]
        return sample


    def interpolate(self, scene_feat, interpolate_len=5):
        eps_start = torch.randn([1, self.n_bps_feat, self.eps_d], dtype=torch.float32).to(device)
        eps_end = torch.randn([1, self.n_bps_feat, self.eps_d], dtype=torch.float32).to(device)
        eps_list = [eps_start]

        for i in range(interpolate_len):
            cur_eps = eps_start + (i+1) * (eps_end - eps_start) / (interpolate_len+1)
            eps_list.append(cur_eps)
        eps_list.append(eps_end)

        gen_list = []
        for eps in eps_list:
            x = F.relu(self.bn3(self.fc3(eps)))  # eps_d-->512, [bs, 1, 512]
            x = F.relu(self.bn_resblc3(self.fc_resblc3(torch.cat([x, scene_feat[0]], dim=-1))))  # 1024-->512
            x = self.res_block3(x)  # [bs, 1, 512]
            x = F.relu(self.bn_resblc4(self.fc_resblc4(torch.cat([x, scene_feat[1]], dim=-1))))  # 1024-->512
            x = self.res_block4(x)  # [bs, 1, 512]
            x = F.relu(self.bn4(self.fc4(torch.cat([x, scene_feat[2]], dim=-1))))  # 1024-->1024 [bs, 1, 1024]
            sample = F.relu(self.bn5(self.fc5(x)))  # 1024 --> 10000, [bs, 1, 10000]
            gen_list.append(sample)
        return gen_list





# body regressor: (body bps, scene bps verts mid_feat) --> body verts/params
class Body_Dec_shift(nn.Module):
    def __init__(self, n_bps=10000, n_bps_feat=1, hsize1=1024,  hsize2=512,
                 n_body_verts=10475, body_param_dim=75, rec_goal='body_verts'):
        super(Body_Dec_shift, self).__init__()
        self.n_body_verts = n_body_verts
        self.rec_goal = rec_goal

        self.bn0 = nn.BatchNorm1d(n_bps_feat)
        self.fc1 = nn.Linear(in_features=n_bps, out_features=hsize1)
        self.bn1 = nn.BatchNorm1d(n_bps_feat)

        self.fc2 = nn.Linear(in_features=hsize1 + hsize2, out_features=hsize1)
        self.bn2 = nn.BatchNorm1d(n_bps_feat)
        if self.rec_goal == 'body_verts':
            self.fc3_verts = nn.Linear(in_features=n_bps_feat * hsize1, out_features=n_body_verts * 3)
        elif self.rec_goal == 'body_params':
            self.fc3 = nn.Linear(in_features=n_bps_feat * hsize1, out_features=body_param_dim)

        self.fc3_shift = nn.Linear(in_features=n_bps_feat * hsize1, out_features=3)

        self.dropout = nn.Dropout(0.5)


    def forward(self, body_bps, scene_bps_verts_feat):
        x = self.bn0(body_bps)
        x = F.relu(self.bn1(self.fc1(x)))  # [bs, n_bps_feat, hsize1]
        x = torch.cat([x, scene_bps_verts_feat], dim=-1)  # [bs, n_bps_feat, hsize1+hsize2]
        x = F.relu(self.bn2(self.fc2(x)))  # [bs, n_bps_feat, hsize1]
        x = x.view(x.shape[0], -1)  # [bs, n_bps_feat * hsize1]

        if self.rec_goal == 'body_verts':
            body_rec = self.fc3_verts(x)  # [bs, n_body_verts*3]
            body_rec = body_rec.view(-1, 3, self.n_body_verts)  # [bs, 3, body_verts]
        elif self.rec_goal == 'body_params':
            body_rec = self.dropout(x)
            body_rec = self.fc3(body_rec)  # [bs, 75]
        # learnt shifting for contact loss
        shift = self.fc3_shift(x)  # [bs, 3]
        return body_rec, shift





# AE for scene bps verts
class Verts_AE(nn.Module):
    def __init__(self, n_bps=10000, hsize1=1024,  hsize2=512):
        super(Verts_AE, self).__init__()
        self.n_bps = n_bps

        # encoder
        self.bn0 = nn.BatchNorm1d(3)  # normalize over last axis (n_bps)
        self.fc1 = nn.Linear(in_features=n_bps, out_features=hsize1)
        self.bn1 = nn.BatchNorm1d(3)
        self.fc2 = nn.Linear(in_features=hsize1, out_features=hsize2)
        self.bn2 = nn.BatchNorm1d(3)
        self.fc3 = nn.Linear(in_features=hsize2*3, out_features=hsize2)

        self.bn3 = nn.BatchNorm1d(1)
        self.fc4 = nn.Linear(in_features=hsize2, out_features=hsize2)
        self.bn4 = nn.BatchNorm1d(1)
        self.fc5 = nn.Linear(in_features=hsize2, out_features=hsize1)
        self.bn5 = nn.BatchNorm1d(1)
        self.fc6 = nn.Linear(in_features=hsize1, out_features=n_bps*3)


    def forward(self, x):
        # input x: selected sbcene bps verts coordinates [bs, 3, n_bps]
        x = self.bn0(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))  # [bs, 3, hsize2]
        x = x.view(x.shape[0], 1, -1)      # [bs, 1, 3*hsize2] (if bps_feat=dist --> n_feat=1)
        mid_feat = self.fc3(x)             # [bs, 1, hsize2]

        x = F.relu(self.bn3(mid_feat))
        x = F.relu(self.bn4(self.fc4(x)))   # [bs, 1, hsize2]
        x = F.relu(self.bn5(self.fc5(x)))  # [bs, 1, hsize1]
        x = self.fc6(x)                    # [bs, 1, n_bps*3]
        x_rec = x.view(x.shape[0], 3, self.n_bps)      # [bs, 3, n_bps]
        return x_rec, mid_feat


