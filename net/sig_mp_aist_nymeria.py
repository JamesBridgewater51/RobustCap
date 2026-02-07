import os

if __name__ == '__main__':
    import sys
    # os.chdir('..') # Removed to allow running from root
    sys.path.insert(0, os.getcwd())

import torch
import articulate as art
from articulate.utils.torch import *
from config import *
from torch.nn.functional import relu
from torch.utils.data import DataLoader, ConcatDataset
import tqdm
from articulate.utils.print import *
import random
import argparse

# Define the Nymeria output path since it might not be in config.py yet
# Based on preprocess_rohm_data.py output location
# Or we can just use paths.amass_dir if we stored it there, but instructions imply separate.
# Let's assume user passes or we define a default if not in config.
NYMERIA_DIR = '/home/minghao/src/robotflow/RoHM/third_party/RobustCap/out/Nymeria_smplx_preprocessed.robustcap'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
body_model = art.ParametricModel(paths.smpl_file, device=device)
# body_model_cpu = art.ParametricModel(paths.smpl_file) # Not used in main logic usually
mp_mask = torch.tensor(mp_mask)

class Net(torch.nn.Module):
    """
    The network for the motion prediction.
    Identical to sig_mp.py Net class.
    """
    hidden_size = 512
    conf_range = (0.7, 0.8)
    contact_threshold = 0.7
    smooth = 1
    use_flat_floor = True
    use_reproj_opt = False
    use_vision_updater = True
    use_imu_updater = True
    name = 'sig_mp_nymeria_aist'  # Joint training with AIST + Nymeria
    gravityc = torch.tensor([-0.0029, 0.9980, -0.0273])
    imu_num = 6
    height_threhold = 0.15
    distrance_threshold = 10
    tran_filter_num = 0.05

    live = False
    update_vision_freq = 30
    update_vision_count = 0
    j_temp = None

    def __init__(self):
        super(Net, self).__init__()
        self.rnn2 = RNNWithInit(input_size=Net.imu_num * 3 + Net.imu_num * 9,
                        output_size=23 * 3,
                        hidden_size=Net.hidden_size,
                        num_rnn_layer=2,
                        dropout=0.4)
        self.rnn3 = RNN(input_size=Net.imu_num * 3 + Net.imu_num * 9 + 23 * 3,
                        output_size=3,
                        hidden_size=Net.hidden_size,
                        num_rnn_layer=2,
                        dropout=0.4)
        self.rnn4 = RNN(input_size=Net.imu_num * 3 + Net.imu_num * 9 + 33 * 3,
                        output_size=23 * 3,
                        hidden_size=1024+256,
                        num_rnn_layer=2,
                        dropout=0.4)
        self.rnn6 = RNN(input_size=Net.imu_num * 3 + Net.imu_num * 9 + 33 * 3 + 23 * 3,
                        output_size=3,
                        hidden_size=1024,
                        num_rnn_layer=2,
                        dropout=0.4)
        self.rnn7 = RNN(input_size=Net.imu_num * 3 + Net.imu_num * 9 + 23 * 3,
                        output_size=24 * 6,
                        hidden_size=512,
                        num_rnn_layer=2,
                        dropout=0.1)
        self.rnn8 = RNN(input_size=Net.imu_num * 3 + Net.imu_num * 9 + 23 * 3,
                        output_size=2,
                        hidden_size=Net.hidden_size,
                        num_rnn_layer=2,
                        dropout=0.4)

        j = body_model.get_zero_pose_joint_and_vertex()[0]
        self.b = body_model.joint_position_to_bone_vector(j.unsqueeze(0)).view(24, 3, 1).to(device)
        self.hidden = [None for _ in range(8)]
        self.temp_hidden = None
        self.last_pfoot = None
        self.last_tran = None
        self.floor_y = []
        self.first_reach = True
        if self.live:
            self.conf_range = (0.85, 0.9)
            self.tran_filter_num = 0.01

    def reset_states(self):
        self.hidden = [None for _ in range(8)]
        self.last_pfoot = None
        self.last_tran = None
        self.floor_y = []
        self.temp_hidden = None
        self.first_reach = True

    @staticmethod
    def cat(*x):
        return [torch.cat(_, dim=1) for _ in zip(*x)]

    @torch.no_grad()
    def forward_online(self, j2dc, accc, oric, first_tran=None, first_frame=False):
        # ... (Identical forward logic as sig_mp.py) ...
        # For brevity, copying the logic exactly is best to avoid subtle bugs
        
        def cat(*x):
            return torch.cat([_.flatten(0) for _ in x]).view(1, -1)

        def f(i, x):  # rnn_i(x)
            rnn = self.__getattr__('rnn%d' % (i + 1))
            x, self.hidden[i] = rnn.rnn(relu(rnn.linear1(x), inplace=True).unsqueeze(1), self.hidden[i])
            return rnn.linear2(x.squeeze(1))

        def fk(glb_pose):
            glb_pose = glb_pose.view(1, 24, 3, 3)
            pb = torch.stack([glb_pose[:, body_model.parent[i]].matmul(self.b[i]) for i in range(1, 24)], dim=1)
            pb = torch.cat((torch.zeros(glb_pose.shape[0], 1, 3, device=device), pb.squeeze(-1)), dim=1)
            return body_model.bone_vector_to_joint_position(pb)[0]

        # visual confidence
        c = j2dc[:, -1].mean().item()
        Rcr = oric.view(Net.imu_num, 3, 3)[-1]

        # inertial
        accr = accc.clone().view(Net.imu_num, 3).mm(Rcr)
        orir = Rcr.t().matmul(oric.clone().view(Net.imu_num, 3, 3))
        j3dr_i = f(1, cat(accr, orir))
        vr = f(2, cat(accr, orir, j3dr_i))

        # visual
        j2dc_clone = j2dc.clone().to(j2dc.device)
        if c > self.conf_range[0] or first_frame:
            j2dc_clone[:, :2] = j2dc_clone[:, :2] / (get_bbox_scale(j2dc_clone))
            j2dc_clone[24:, :2] = j2dc_clone[24:, :2] - j2dc_clone[23:24, :2]
            j2dc_clone[:23, :2] = j2dc_clone[:23, :2] - j2dc_clone[23:24, :2]
            j3dc = f(3, cat(accc, oric, j2dc_clone))
            j3dr_v = j3dc.view(23, 3).mm(Rcr)
            if first_frame:
                pc = f(5, cat(accc, oric, j2dc, j3dc))

        # lerp
        if c >= self.conf_range[1]:
            j3dr = j3dr_v
            pc = f(5, cat(accc, oric, j2dc, j3dc))
        elif c > self.conf_range[0]:
            k = (c - self.conf_range[0]) / (self.conf_range[1] - self.conf_range[0])
            j3dr = art.math.lerp(j3dr_i.view(-1), j3dr_v.view(-1), k)
            pc = f(5, cat(accc, oric, j2dc, j3dc))
        else:
            j3dr = j3dr_i

        poseg6d = f(6, cat(accr, orir, j3dr))
        contact = f(7, cat(accr, orir, j3dr)).sigmoid()

        # pose
        poseg = art.math.r6d_to_rotation_matrix(poseg6d).view(1, 24, 3, 3)
        pose = body_model.inverse_kinematics_R(poseg)[0]
        pose[0] = Rcr

        # update inertial hidden states
        if c >= self.conf_range[1] and self.use_imu_updater and self.first_reach:
            self.first_reach = False
            nd, nh = 2, self.hidden_size
            rnn = self.__getattr__('rnn2')
            h_init, c_init = rnn.init_net(j3dr.flatten()).view(-1, 2, nd, nh).permute(1, 2, 0, 3)
            self.hidden[1] = (h_init, c_init)

        # tran
        pfoot = fk(poseg)[10:12].mm(Rcr.t())
        if contact.max() < self.contact_threshold or self.last_pfoot is None:
            v = Rcr.mm(vr.view(3, 1)).view(3) * vel_scale / 60
        else:
            v = (self.last_pfoot - pfoot)[contact.argmax().item()]
        if self.last_tran is None:
            tran = v
        else:
            tran = self.last_tran.to(device) + v

        if self.conf_range[1] <= c:
            k = (c - self.conf_range[0]) / (self.conf_range[1] - self.conf_range[0])
            if k > 1:
                k = 1
            if (pc - tran).norm() > self.distrance_threshold or self.tran_filter_num > 1:
                tran = pc
            else:
                tran = art.math.lerp(tran, pc, self.tran_filter_num * k)

        # determine floor height
        tran = tran.view(3)
        self.gravityc = self.gravityc.to(device)
        if len(self.floor_y) < 11 and not first_frame and first_tran is None and contact.max() > self.contact_threshold and self.use_flat_floor and c >= self.conf_range[1]:
            p0 = torch.dot(pfoot[0] + tran.to(device), self.gravityc) * self.gravityc
            p1 = torch.dot(pfoot[1] + tran.to(device), self.gravityc) * self.gravityc
            if p0.norm() < p1.norm():
                self.floor_y.append(p1)
            else:
                self.floor_y.append(p0)
        if self.use_flat_floor and len(self.floor_y) > 10 and contact.max() > self.contact_threshold:
            p0 = torch.dot(pfoot[0] + tran.to(device), self.gravityc) * self.gravityc
            p1 = torch.dot(pfoot[1] + tran.to(device), self.gravityc) * self.gravityc
            if p0.norm() < p1.norm() and (sum(self.floor_y[-6:]) / 6 - p1).norm() < self.height_threhold:
                tran = tran + (sum(self.floor_y[-6:]) / 6 - p1)
            elif (sum(self.floor_y[-6:]) / 6 - p0).norm() < self.height_threhold:
                tran = tran + (sum(self.floor_y[-6:]) / 6 - p0)
        if first_tran is not None:
            tran = first_tran.to(device)
        elif first_frame:
            tran = pc.to(device)

        self.last_pfoot = pfoot
        if self.use_reproj_opt or self.use_vision_updater:
            if not self.live:
                grot, joint, vert = body_model.forward_kinematics(pose.view(-1, 24, 3, 3), tran=tran.view(-1, 3).to(device),
                                                              calc_mesh=True)
                j = sync_mp3d(vert[0], joint[0])
            else:
                if self.update_vision_count == 0:
                    grot, joint, vert = body_model.forward_kinematics(pose.view(-1, 24, 3, 3), tran=tran.view(-1, 3).to(device),
                                                                      calc_mesh=True)
                    j = sync_mp3d(vert[0], joint[0])
                    self.j_temp = j
                    self.update_vision_count = self.update_vision_freq
                else:
                    j = self.j_temp
                    self.update_vision_count -= 1

        # optimize reprojection error
        if self.use_reproj_opt and c > self.conf_range[0]:
            p = j2dc[:, 2]
            ax = (p / j[:, 2].pow(2)).sum() + self.smooth
            bx = (p * (- j[:, 0] / j[:, 2].pow(2) + j2dc[:, 0] / j[:, 2])).sum()
            ay = (p / j[:, 2].pow(2)).sum() + self.smooth
            by = (p * (- j[:, 1] / j[:, 2].pow(2) + j2dc[:, 1] / j[:, 2])).sum()
            d_tran = torch.tensor([bx / ax, by / ay, 0]).to(device)
            tran = tran + d_tran
            j = j + d_tran
            az = (p * (j[:, 0].pow(2) + j[:, 1].pow(2)) / j[:, 2].pow(4)).sum() + self.smooth
            bz = (p * ((j[:, 0] / j[:, 2] - j2dc[:, 0]) * j[:, 0] / j[:, 2].pow(2) + (
                    j[:, 1] / j[:, 2] - j2dc[:, 1]) * j[:, 1] / j[:, 2].pow(2))).sum()
            d_tran = torch.tensor([0, 0, bz / az]).to(device)
            tran = tran + d_tran
            j = j + d_tran.to(device)

        # visual forward for hidden states
        if self.use_vision_updater and c <= self.conf_range[0] and (self.update_vision_count == self.update_vision_freq or not self.live):
            j2dc = j / j[:, 2:]
            j3dc = joint[0][1:] - joint[0][:1]
            f(5, cat(accc, oric, j2dc, j3dc))
            j2dc[:, :2] = j2dc[:, :2] / get_bbox_scale(j2dc)
            j2dc[24:, :2] = j2dc[24:, :2] - j2dc[23:24, :2]
            j2dc[:23, :2] = j2dc[:23, :2] - j2dc[23:24, :2]
            f(3, cat(accc, oric, j2dc)).view(23, 3)

        self.last_tran = tran
        return pose.view(24, 3, 3), tran.view(3)

def get_bbox_scale(uv):
    u_max, u_min = uv[..., 0].max(dim=-1).values, uv[..., 0].min(dim=-1).values
    v_max, v_min = uv[..., 1].max(dim=-1).values, uv[..., 1].min(dim=-1).values
    return torch.max(u_max - u_min, v_max - v_min)

def sync_mp3d(vert, joint):
    syn_3d = vert[mp_mask]
    syn_3d[11:17] = joint[16:22].clone()
    syn_3d[23:25] = joint[1:3].clone()
    syn_3d[25:27] = joint[4:6].clone()
    syn_3d[27:29] = joint[7:9].clone()
    return syn_3d


# ==================================================================================
# Nymeria Dataset & Training Loops
# ==================================================================================

# Helper for loading Nymeria data consistently for all RNN stages
def load_nymeria_dataset(data_dir, kind, split_size=-1):
    path = os.path.join(data_dir, f'{kind}_nymeria.pt')
    print(f'Reading {kind} dataset from {path}')
    return torch.load(path)

def train_rnn2():
    def AISTDataset(data_dir, kind, split_size=-1):
        """
        AIST dataset loader for RNN2 (from sig_mp.py)
        """
        print('Reading %s dataset "%s"' % (kind, data_dir))
        dataset = torch.load(os.path.join(data_dir, kind + '.pt'))
        data, label = [], []
        for i in tqdm.trange(len(dataset['pose'])):  # ith sequence
            Rrw = art.math.axis_angle_to_rotation_matrix(dataset['pose'][i][:, :3]).transpose(1, 2)
            orir = Rrw.unsqueeze(1).matmul(dataset['imu_ori'][i])
            accr = Rrw.unsqueeze(1).matmul(dataset['imu_acc'][i].unsqueeze(-1)).squeeze(-1)
            j3dr = Rrw.unsqueeze(1).matmul(dataset['joint3d'][i].unsqueeze(-1)).squeeze(-1)
            j3dr = j3dr[:, 1:] - j3dr[:, :1]
            data.append(torch.cat((accr.flatten(1), orir.flatten(1)), dim=1)[1:-1])
            label.append(j3dr.flatten(1)[1:-1])
        return RNNWithInitDataset(data, label, split_size=split_size, device=device)

    def NymeriaDataset(data_dir, kind, split_size=-1):
        dataset = load_nymeria_dataset(data_dir, kind)
        data, label = [], []
        for i in tqdm.trange(len(dataset['imu_acc'])):
            p = art.math.axis_angle_to_rotation_matrix(dataset['pose'][i]).view(-1, 24, 3, 3)
            j3dr = (dataset['joint3d'][i][:, 1:] - dataset['joint3d'][i][:, :1]).bmm(p[:, 0])
            accw = dataset['imu_acc'][i]
            oriw = dataset['imu_ori'][i]
            Rrw = p[:, 0].transpose(1, 2)
            accr = Rrw.unsqueeze(1).matmul(accw.unsqueeze(-1))
            orir = Rrw.unsqueeze(1).matmul(oriw)
            data.append(torch.cat((accr.flatten(1), orir.flatten(1)), dim=1)[1:-1])
            label.append(j3dr.flatten(1)[1:-1])
        return RNNWithInitDataset(data, label, split_size=split_size, device=device)

    print_yellow('=================== Training RNN2 [AIST + Nymeria] ===================')
    rnn_mse_loss_fn = RNNLossWrapper(torch.nn.MSELoss())
    rnn_dist_eval_fn = RNNLossWrapper(art.PositionErrorEvaluator())
    save_dir = os.path.join(paths.weight_dir, Net.name, 'rnn2')
    net = Net().rnn2.to(device)

    train_dataloader = DataLoader(
        ConcatDataset([
            AISTDataset(paths.aist_dir, kind='train', split_size=200),
            NymeriaDataset(NYMERIA_DIR, kind='train', split_size=200)
        ]),
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=RNNDataset.make_collate_fn(device)
    )
    valid_dataloader = DataLoader(
        ConcatDataset([
            AISTDataset(paths.aist_dir, kind='val', split_size=200),
            NymeriaDataset(NYMERIA_DIR, kind='val', split_size=200)
        ]),
        batch_size=64,
        num_workers=2,
        pin_memory=True,
        collate_fn=RNNDataset.make_collate_fn(device)
    )

    train(net, train_dataloader, valid_dataloader, save_dir, loss_fn=rnn_mse_loss_fn, eval_fn=rnn_dist_eval_fn,
          num_epoch=150, num_iter_between_vald=20, clip_grad_norm=1, load_last_states=True,
          eval_metric_names=['distance error (m)'], wandb_project_name='sig_mp_nymeria_aist',
          wandb_config=None, wandb_watch=True, wandb_name='rnn2')

def train_rnn3():
    def augment_fn(x):
        x = x.clone()
        x[:, -69:] = torch.normal(x[:, -69:], 0.04)
        return x

    def AISTDataset(data_dir, kind, split_size=-1):
        """
        AIST dataset loader for RNN3 (from sig_mp.py)
        """
        print('Reading %s dataset "%s"' % (kind, data_dir))
        dataset = torch.load(os.path.join(data_dir, kind + '.pt'))
        data, label = [], []
        for i in tqdm.trange(len(dataset['pose'])):  # ith sequence
            Rrw = art.math.axis_angle_to_rotation_matrix(dataset['pose'][i][:, :3]).transpose(1, 2)
            orir = Rrw.unsqueeze(1).matmul(dataset['imu_ori'][i])
            accr = Rrw.unsqueeze(1).matmul(dataset['imu_acc'][i].unsqueeze(-1)).squeeze(-1)
            j3dr = Rrw.unsqueeze(1).matmul(dataset['joint3d'][i].unsqueeze(-1)).squeeze(-1)
            j3dr = j3dr[:, 1:] - j3dr[:, :1]
            v3dw = (dataset['joint3d'][i][2:] - dataset['joint3d'][i][:-2]) * 30
            v3dw = torch.cat((torch.zeros(1, 3), v3dw[:, 0], torch.zeros(1, 3)), dim=0) / vel_scale
            v3dr = Rrw.matmul(v3dw.unsqueeze(-1)).squeeze(-1)
            data.append(torch.cat((accr.flatten(1), orir.flatten(1), j3dr.flatten(1)), dim=1)[1:-1])
            label.append(v3dr.flatten(1)[1:-1])
        return RNNDataset(data, label, split_size=split_size, augment_fn=augment_fn, device=device)

    def NymeriaDataset(data_dir, kind, split_size=-1):
        dataset = load_nymeria_dataset(data_dir, kind)
        data, label = [], []
        for i in tqdm.trange(len(dataset['imu_acc'])):
            p = art.math.axis_angle_to_rotation_matrix(dataset['pose'][i]).view(-1, 24, 3, 3)
            j3dr = (dataset['joint3d'][i][:, 1:] - dataset['joint3d'][i][:, :1]).bmm(p[:, 0])
            accw = dataset['imu_acc'][i]
            oriw = dataset['imu_ori'][i]
            Rrw = p[:, 0].transpose(1, 2)
            accr = Rrw.unsqueeze(1).matmul(accw.unsqueeze(-1))
            orir = Rrw.unsqueeze(1).matmul(oriw)
            v3dw = (dataset['joint3d'][i][2:] - dataset['joint3d'][i][:-2]) * 30
            v3dw = torch.cat((torch.zeros(1, 3), v3dw[:, 0], torch.zeros(1, 3)), dim=0) / vel_scale
            v3dr = Rrw.matmul(v3dw.unsqueeze(-1)).squeeze(-1)
            data.append(torch.cat((accr.flatten(1), orir.flatten(1), j3dr.flatten(1)), dim=1)[1:-1])
            label.append(v3dr.flatten(1)[1:-1])
        return RNNDataset(data, label, split_size=split_size, augment_fn=augment_fn, device=device)

    print_yellow('=================== Training RNN3 [AIST + Nymeria] ===================')
    def loss_fn(x, y):
        l = x.shape[0]
        f1 = mse(x, y)
        f6 = mse(x[l % 6:].view(-1, 6, 3).sum(dim=1), y[l % 6:].view(-1, 6, 3).sum(dim=1))
        f20 = mse(x[l % 20:].view(-1, 20, 3).sum(dim=1), y[l % 20:].view(-1, 20, 3).sum(dim=1))
        f60 = mse(x[l % 60:].view(-1, 60, 3).sum(dim=1), y[l % 60:].view(-1, 60, 3).sum(dim=1))
        return f1 + f6 + f20 + f60

    mse = torch.nn.MSELoss()
    rnn_loss_fn = RNNLossWrapper(loss_fn)
    save_dir = os.path.join(paths.weight_dir, Net.name, 'rnn3')
    net = Net().rnn3.to(device)

    train_dataloader = DataLoader(
        ConcatDataset([
            AISTDataset(paths.aist_dir, kind='train', split_size=200),
            NymeriaDataset(NYMERIA_DIR, kind='train', split_size=200)
        ]),
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=RNNDataset.make_collate_fn(device)
    )
    valid_dataloader = DataLoader(
        ConcatDataset([
            AISTDataset(paths.aist_dir, kind='val', split_size=200),
            NymeriaDataset(NYMERIA_DIR, kind='val', split_size=200)
        ]),
        batch_size=64,
        num_workers=2,
        pin_memory=True,
        collate_fn=RNNDataset.make_collate_fn(device)
    )

    train(net, train_dataloader, valid_dataloader, save_dir, loss_fn=rnn_loss_fn, eval_fn=rnn_loss_fn,
          num_epoch=200, num_iter_between_vald=20, clip_grad_norm=1, load_last_states=True,
          wandb_project_name='sig_mp_nymeria_aist',
          wandb_config=None, wandb_watch=True, wandb_name='rnn3')


def train_rnn4():

    def augment_fn(x):
        return x

    def AISTDataset(data_dir, kind, split_size=-1):
        """
        AIST dataset loader for RNN4 (from sig_mp.py) - uses real camera and 2D keypoints
        """
        print('Reading %s dataset "%s"' % (kind, data_dir))
        dataset = torch.load(os.path.join(data_dir, kind + '.pt'))
        data, label = [], []
        for i in tqdm.trange(len(dataset['pose'])):  # ith sequence
            for j in range(9):  # jth camera view
                if dataset['joint2d_mp'][i][j] is None: continue
                Tcw = dataset['cam_T'][i][j]
                Kinv = dataset['cam_K'][i][j].inverse()
                oric = Tcw[:3, :3].matmul(dataset['imu_ori'][i])
                accc = Tcw.matmul(art.math.append_zero(dataset['imu_acc'][i]).unsqueeze(-1)).squeeze(-1)[..., :3]
                j3dc = Tcw.matmul(art.math.append_one(dataset['joint3d'][i]).unsqueeze(-1)).squeeze(-1)[..., :3]
                j3dc = j3dc[:, 1:] - j3dc[:, :1]
                j2dc = torch.zeros(len(oric), 33, 3)
                j2dc[..., :2] = dataset['joint2d_mp'][i][j][..., :2]
                j2dc[..., 0] = j2dc[..., 0] * 1920
                j2dc[..., 1] = j2dc[..., 1] * 1080
                j2dc = Kinv.matmul(art.math.append_one(j2dc[..., :2]).unsqueeze(-1)).squeeze(-1)
                j2dc[..., :2] = j2dc[..., :2] / (get_bbox_scale(j2dc)).view(-1, 1, 1)
                j2dc[:, 24:, :2] = j2dc[:, 24:, :2] - j2dc[:, 23:24, :2]
                j2dc[:, :23, :2] = j2dc[:, :23, :2] - j2dc[:, 23:24, :2]
                j2dc[..., -1] = dataset['joint2d_mp'][i][j][..., -1]
                data.append(torch.cat((accc.flatten(1), oric.flatten(1), j2dc.flatten(1)), dim=1)[1:-1])
                label.append(j3dc.flatten(1)[1:-1])
        return RNNDataset(data, label, split_size=split_size, device=device, augment_fn=augment_fn)

    class NymeriaDataset(RNNDataset):
        def __init__(self, data_dir, kind, split_size=-1):
            dataset = load_nymeria_dataset(data_dir, kind)
            data, label = [], []
            # We need the confidence prior from synthetic data? 
            # RobusCap source: self.conf = torch.load('data/dataset_work/syn_c.pt')
            # Assuming this file exists and is relevant.
            self.conf = torch.load('data/dataset_work/syn_c.pt')

            for i in tqdm.trange(len(dataset['imu_acc'])):
                accw = dataset['imu_acc'][i]
                oriw = dataset['imu_ori'][i]
                root = dataset['joint3d'][i][0, 0].clone()
                j3dw = dataset['joint3d'][i] - root
                j3dw_mp = dataset['sync_3d_mp'][i] - root
                # Map SMPL joints to extended MP joints (same mapping as AMASSDataset)
                j3dw_mp[:, 11] = j3dw[:, 16].clone()
                j3dw_mp[:, 12] = j3dw[:, 17].clone()
                j3dw_mp[:, 13] = j3dw[:, 18].clone()
                j3dw_mp[:, 14] = j3dw[:, 19].clone()
                j3dw_mp[:, 15] = j3dw[:, 20].clone()
                j3dw_mp[:, 16] = j3dw[:, 21].clone()
                j3dw_mp[:, 23] = j3dw[:, 1].clone()
                j3dw_mp[:, 24] = j3dw[:, 2].clone()
                j3dw_mp[:, 25] = j3dw[:, 4].clone()
                j3dw_mp[:, 26] = j3dw[:, 5].clone()
                j3dw_mp[:, 27] = j3dw[:, 7].clone()
                j3dw_mp[:, 28] = j3dw[:, 8].clone()
                if split_size > 0:
                    T = j3dw.shape[0]
                    for start in range(0, T, split_size):
                        end = min(start + split_size, T)
                        if end - start < 10:  # Skip very short chunks
                            continue
                        
                        # Chunk slice
                        accw_chunk = accw[start:end]
                        oriw_chunk = oriw[start:end]
                        j3dw_mp_chunk = j3dw_mp[start:end]
                        j3dw_chunk = j3dw[start:end]
                        
                        # Re-center relative to the START of this chunk
                        chunk_root_offset = j3dw_chunk[0, 0].clone() 
                        j3dw_chunk = j3dw_chunk - chunk_root_offset
                        j3dw_mp_chunk = j3dw_mp_chunk - chunk_root_offset
                        
                        data.append(torch.cat((accw_chunk.flatten(1), oriw_chunk.flatten(1), j3dw_mp_chunk.flatten(1)), dim=1)[1:-1])
                        label.append(j3dw_chunk.flatten(1)[1:-1])
                else:
                    data.append(torch.cat((accw.flatten(1), oriw.flatten(1), j3dw_mp.flatten(1)), dim=1)[1:-1])
                    label.append(j3dw.flatten(1)[1:-1])
            super(NymeriaDataset, self).__init__(data, label, split_size=-1)

        def __getitem__(self, i):
            data, label = super(NymeriaDataset, self).__getitem__(i)
            accw = data[:, :18].reshape(-1, 6, 3, 1)
            oriw = data[:, 18:18+6*3*3].reshape(-1, 6, 3, 3)
            j3dw_mp = data[:, -33*3:].reshape(-1, 33, 3, 1)
            j3dw = label.reshape(-1, 24, 3, 1)

            # Synthetic Camera Projection Logic (Identical to AMASSDataset)
            Rwc0 = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1.]])
            Rc0c = art.math.generate_random_rotation_matrix_constrained(n=1, y=(-90, 90), p=(-30, 30), r=(-5, 5))[0]
            Rcw = Rwc0.mm(Rc0c).t()

            accc = Rcw.matmul(accw)
            oric = Rcw.matmul(oriw)
            j3dc = Rcw.matmul(j3dw).squeeze(-1)
            j3dc_mp = Rcw.matmul(j3dw_mp).squeeze(-1)

            random_tranc = art.math.lerp(torch.tensor([-1, -1, 3.]), torch.tensor([1, 1, 8.]), torch.rand(3))
            random_tranc[2] -= j3dc[..., -1].min()
            j3dc = j3dc + random_tranc
            j3dc_mp = j3dc_mp + random_tranc
            j2dc = j3dc_mp / j3dc_mp[..., -1:]
            
            ran = range(0, len(self.conf))
            rand = random.sample(ran, len(accc))
            p = self.conf[rand]
            j2dc[..., :2] = torch.normal(j2dc[..., :2], 0.003 * (1 - p))
            j2dc[..., -1:] = p
            j2dc[..., :2] = j2dc[..., :2] / (get_bbox_scale(j2dc)).view(-1, 1, 1)
            j2dc[:, 24:, :2] = j2dc[:, 24:, :2] - j2dc[:, 23:24, :2]
            j2dc[:, :23, :2] = j2dc[:, :23, :2] - j2dc[:, 23:24, :2]
            j3dc = j3dc[:, 1:] - j3dc[:, :1]
            # RobustCap original code: data = torch.cat((accc.flatten(1), oric.flatten(1), j2dc.flatten(1), j3dc.flatten(1)), dim=1)
            # But we should not have j3dc in the input, since
            # 输入语义应该是 IMU Data: 包含身体 6 个部位的惯性测量数据（加速度计 + 旋转矩阵），已统一转换到虚拟相机坐标系下 (Rcw 变换)。 + 2D Keypoints: 33 个关节点的 2D 投影坐标及置信度。经过了归一化、噪声注入和根节点相对化处理.
            # 输出语义是 3D Pose: 目标是预测 虚拟相机坐标系 下的 根节点相对 (Root-Relative) 3D 关节位置。包含 23 个关节（排除了根节点 0）.
            # 输出语义不应该 包含 j3dc
            data = torch.cat((accc.flatten(1), oric.flatten(1), j2dc.flatten(1)), dim=1)
            label = j3dc.flatten(1)
            return augment_fn(data), label  # GPU transfer happens in collate_fn

    print_yellow('=================== Training RNN4 [AIST + Nymeria] ===================')
    rnn_mse_loss_fn = RNNLossWrapper(torch.nn.MSELoss())
    rnn_dist_eval_fn = RNNLossWrapper(art.PositionErrorEvaluator())
    save_dir = os.path.join(paths.weight_dir, Net.name, 'rnn4')
    net = Net().rnn4.to(device)
    
    train_dataloader = DataLoader(
        ConcatDataset([
            AISTDataset(paths.aist_dir, kind='train', split_size=200),
            NymeriaDataset(NYMERIA_DIR, kind='train', split_size=200)
        ]),
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=RNNDataset.make_collate_fn(device)
    )
    valid_dataloader = DataLoader(
        ConcatDataset([
            AISTDataset(paths.aist_dir, kind='val', split_size=200),
            NymeriaDataset(NYMERIA_DIR, kind='val', split_size=200)
        ]),
        batch_size=64,
        num_workers=2,
        pin_memory=True,
        collate_fn=RNNDataset.make_collate_fn(device)
    )
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    train(net, train_dataloader, valid_dataloader, save_dir, loss_fn=rnn_mse_loss_fn, eval_fn=rnn_dist_eval_fn,
          num_epoch=200, num_iter_between_vald=60, clip_grad_norm=1, load_last_states=True,
          eval_metric_names=['distance error (m)'], wandb_project_name='sig_mp_nymeria_aist',
          wandb_config=None, wandb_watch=True, wandb_name='rnn4_final', optimizer=optimizer)

def train_rnn6():
    def augment_fn(x):
        x = x.clone()
        x[:, -69:] = torch.normal(x[:, -69:], 0.03)
        return x

    class NymeriaDataset(RNNDataset):
        def __init__(self, data_dir, kind, split_size=-1):
            dataset = load_nymeria_dataset(data_dir, kind)
            self.conf = torch.load('data/dataset_work/syn_c.pt')
            data, label = [], []
            for i in tqdm.trange(len(dataset['imu_acc'])):
                accw = dataset['imu_acc'][i]
                oriw = dataset['imu_ori'][i]
                root = dataset['joint3d'][i][0, 0].clone()
                j3dw = dataset['joint3d'][i] - root
                j3dw_mp = dataset['sync_3d_mp'][i] - root
                # Mapping
                j3dw_mp[:, 11] = j3dw[:, 16].clone()
                j3dw_mp[:, 12] = j3dw[:, 17].clone()
                j3dw_mp[:, 13] = j3dw[:, 18].clone()
                j3dw_mp[:, 14] = j3dw[:, 19].clone()
                j3dw_mp[:, 15] = j3dw[:, 20].clone()
                j3dw_mp[:, 16] = j3dw[:, 21].clone()
                j3dw_mp[:, 23] = j3dw[:, 1].clone()
                j3dw_mp[:, 24] = j3dw[:, 2].clone()
                j3dw_mp[:, 25] = j3dw[:, 4].clone()
                j3dw_mp[:, 26] = j3dw[:, 5].clone()
                j3dw_mp[:, 27] = j3dw[:, 7].clone()
                j3dw_mp[:, 28] = j3dw[:, 8].clone()
                if split_size > 0:
                    T = j3dw.shape[0]
                    for start in range(0, T, split_size):
                        end = min(start + split_size, T)
                        if end - start < 10:  # Skip very short chunks
                            continue
                        
                        # Chunk slice
                        accw_chunk = accw[start:end]
                        oriw_chunk = oriw[start:end]
                        j3dw_mp_chunk = j3dw_mp[start:end]
                        j3dw_chunk = j3dw[start:end]
                        
                        # Re-center relative to the START of this chunk
                        chunk_root_offset = j3dw_chunk[0, 0].clone() 
                        j3dw_chunk = j3dw_chunk - chunk_root_offset
                        j3dw_mp_chunk = j3dw_mp_chunk - chunk_root_offset
                        
                        data.append(torch.cat((accw_chunk.flatten(1), oriw_chunk.flatten(1), j3dw_mp_chunk.flatten(1)), dim=1)[1:-1])
                        label.append(j3dw_chunk.flatten(1)[1:-1])
                else:
                    data.append(torch.cat((accw.flatten(1), oriw.flatten(1), j3dw_mp.flatten(1)), dim=1)[1:-1])
                    label.append(j3dw.flatten(1)[1:-1])
            super(NymeriaDataset, self).__init__(data, label, split_size=-1)

        def __getitem__(self, i):
            data, label = super(NymeriaDataset, self).__getitem__(i)
            accw = data[:, :18].reshape(-1, 6, 3, 1)
            oriw = data[:, 18:18 + 6 * 3 * 3].reshape(-1, 6, 3, 3)
            j3dw_mp = data[:, -33 * 3:].reshape(-1, 33, 3, 1)
            j3dw = label.reshape(-1, 24, 3, 1)

            Rwc0 = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1.]])
            Rc0c = art.math.generate_random_rotation_matrix_constrained(n=1, y=(-90, 90), p=(-30, 30), r=(-5, 5))[0]
            Rcw = Rwc0.mm(Rc0c).t()

            accc = Rcw.matmul(accw)
            oric = Rcw.matmul(oriw)
            j3dc = Rcw.matmul(j3dw).squeeze(-1)
            j3dc_mp = Rcw.matmul(j3dw_mp).squeeze(-1)

            random_tranc = art.math.lerp(torch.tensor([-1, -1, 3.]), torch.tensor([1, 1, 8.]), torch.rand(3))
            random_tranc[2] -= j3dc[..., -1].min()
            j3dc = j3dc + random_tranc
            j3dc_mp = j3dc_mp + random_tranc
            j2dc = j3dc_mp / j3dc_mp[..., -1:]
            ran = range(0, len(self.conf))
            rand = random.sample(ran, len(accc))
            p = self.conf[rand]
            j2dc[..., :2] = torch.normal(j2dc[..., :2], 0.003 * (1 - p))
            j2dc[..., -1:] = p
            
            label = j3dc[:, 0] # Output is translation
            j3dc = j3dc[:, 1:] - j3dc[:, :1]
            data = torch.cat((accc.flatten(1), oric.flatten(1), j2dc.flatten(1), j3dc.flatten(1)), dim=1)
            return augment_fn(data), label

    print_yellow('=================== Training RNN6 [AIST + Nymeria] ===================')
    rnn_loss_fn = RNNLossWrapper(torch.nn.MSELoss())
    save_dir = os.path.join(paths.weight_dir, Net.name, 'rnn6')
    net = Net().rnn6.to(device)
    
    train_dataloader = DataLoader(
        ConcatDataset([
            NymeriaDataset(NYMERIA_DIR, kind='train', split_size=200)
        ]),
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=RNNDataset.make_collate_fn(device)
    )
    valid_dataloader = DataLoader(
        ConcatDataset([
            NymeriaDataset(NYMERIA_DIR, kind='val', split_size=200)
        ]),
        batch_size=64,
        num_workers=2,
        pin_memory=True,
        collate_fn=RNNDataset.make_collate_fn(device)
    )

    train(net, train_dataloader, valid_dataloader, save_dir, loss_fn=rnn_loss_fn, eval_fn=rnn_loss_fn,
          num_epoch=100, num_iter_between_vald=60, clip_grad_norm=1, load_last_states=True,
          wandb_project_name='sig_mp_nymeria_aist',
          wandb_config=None, wandb_watch=True, wandb_name='rnn6', lr_scheduler_patience=5)

def train_rnn7():
    def augment_fn(x):
        x = torch.normal(x, 0.03)
        return x

    def AISTDataset(data_dir, kind, split_size=-1):
        """
        AIST dataset loader for RNN7 (from sig_mp.py)
        """
        print('Reading %s dataset "%s"' % (kind, data_dir))
        dataset = torch.load(os.path.join(data_dir, kind + '.pt'))
        data, label = [], []
        for i in tqdm.trange(len(dataset['pose'])):  # ith sequence
            Rrw = art.math.axis_angle_to_rotation_matrix(dataset['pose'][i][:, :3]).transpose(1, 2)
            orir = dataset['imu_ori'][i].clone()
            orir[:, :5] = Rrw.unsqueeze(1).matmul(dataset['imu_ori'][i][:, :5])
            accr = Rrw.unsqueeze(1).matmul(dataset['imu_acc'][i].unsqueeze(-1)).squeeze(-1)
            j3dr = Rrw.unsqueeze(1).matmul(dataset['joint3d'][i].unsqueeze(-1)).squeeze(-1)
            j3dr = j3dr[:, 1:] - j3dr[:, :1]
            pose = art.math.axis_angle_to_rotation_matrix(dataset['pose'][i]).view(-1, 24, 3, 3)
            pose[:, 0] = torch.eye(3)
            pose = art.math.rotation_matrix_to_r6d(body_model.forward_kinematics_R(pose)).view(-1, 24, 6)
            data.append(torch.cat((accr.flatten(1), orir.flatten(1), j3dr.flatten(1)), dim=1)[1:-1])
            label.append(pose.flatten(1)[1:-1])
        return RNNDataset(data, label, split_size=split_size, augment_fn=augment_fn, device=device)

    def NymeriaDataset(data_dir, kind, split_size=-1):
        dataset = load_nymeria_dataset(data_dir, kind)
        data, label = [], []
        for i in tqdm.trange(len(dataset['imu_acc'])):
            p = art.math.axis_angle_to_rotation_matrix(dataset['pose'][i]).view(-1, 24, 3, 3)
            j3dr = (dataset['joint3d'][i][:, 1:] - dataset['joint3d'][i][:, :1]).bmm(p[:, 0])
            accw = dataset['imu_acc'][i]
            oriw = dataset['imu_ori'][i]
            Rrw = p[:, 0].transpose(1, 2)
            accr = Rrw.unsqueeze(1).matmul(accw.unsqueeze(-1))
            orir = oriw.clone()
            orir[:, :5] = Rrw.unsqueeze(1).matmul(oriw[:, :5])
            p[:, 0] = torch.eye(3)
            glbp = body_model.forward_kinematics_R(p)
            p6d = art.math.rotation_matrix_to_r6d(glbp).view(-1, 24 * 6)
            data.append(torch.cat((accr.flatten(1), orir.flatten(1), j3dr.flatten(1)), dim=1)[1:-1])
            label.append(p6d.flatten(1)[1:-1])
        return RNNDataset(data, label, split_size=split_size, augment_fn=augment_fn, device=device)

    class Loss:
        def __init__(self):
            j = body_model.get_zero_pose_joint_and_vertex()[0]
            self.b = body_model.joint_position_to_bone_vector(j.unsqueeze(0)).view(24, 3, 1).to(device)

        @staticmethod
        def weighted_mse(x, y, w=1):
            return ((x - y).pow(2) * w).mean()

        def forward_kinematics(self, p):
            p = art.math.r6d_to_rotation_matrix(p).view(-1, 24, 3, 3)
            pb = torch.stack([p[:, body_model.parent[i]].matmul(self.b[i]) for i in range(1, 24)], dim=1)
            pb = torch.cat((torch.zeros(p.shape[0], 1, 3, device=device), pb.squeeze(-1)), dim=1)
            return body_model.bone_vector_to_joint_position(pb)

        def __call__(self, x, y):
            l1 = self.weighted_mse(x, y)
            l2 = self.weighted_mse(self.forward_kinematics(x), self.forward_kinematics(y))
            return l1 + l2 * 100

    print_yellow('=================== Training RNN7 [AIST + Nymeria] ===================')
    rnn_loss_fn = RNNLossWrapper(Loss())
    save_dir = os.path.join(paths.weight_dir, Net.name, 'rnn7')
    net = Net().rnn7.to(device)
    
    train_dataloader = DataLoader(
        ConcatDataset([
            AISTDataset(paths.aist_dir, kind='train', split_size=200),
            NymeriaDataset(NYMERIA_DIR, kind='train', split_size=200)
        ]),
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=RNNDataset.make_collate_fn(device)
    )
    valid_dataloader = DataLoader(
        ConcatDataset([
            AISTDataset(paths.aist_dir, kind='val', split_size=200),
            NymeriaDataset(NYMERIA_DIR, kind='val', split_size=200)
        ]),
        batch_size=64,
        num_workers=2,
        pin_memory=True,
        collate_fn=RNNDataset.make_collate_fn(device)
    )

    train(net, train_dataloader, valid_dataloader, save_dir, loss_fn=rnn_loss_fn, eval_fn=rnn_loss_fn,
          num_epoch=120, num_iter_between_vald=20, clip_grad_norm=1, load_last_states=True,
          wandb_project_name='sig_mp_nymeria_aist',
          wandb_config=None, wandb_watch=True, wandb_name='rnn7', lr_scheduler_patience=5)

def train_rnn8():

    def augment_fn(x):
        x = x.clone()
        x[:, -69:] = torch.normal(x[:, -69:], 0.03)
        return x

    def AISTDataset(data_dir, kind, split_size=-1):
        """
        AIST dataset loader for RNN8 (from sig_mp.py)
        """
        print('Reading %s dataset "%s"' % (kind, data_dir))
        dataset = torch.load(os.path.join(data_dir, kind + '.pt'))
        data, label = [], []
        for i in tqdm.trange(len(dataset['pose'])):  # ith sequence
            Rrw = art.math.axis_angle_to_rotation_matrix(dataset['pose'][i][:, :3]).transpose(1, 2)
            orir = Rrw.unsqueeze(1).matmul(dataset['imu_ori'][i])
            accr = Rrw.unsqueeze(1).matmul(dataset['imu_acc'][i].unsqueeze(-1)).squeeze(-1)
            j3dr = Rrw.unsqueeze(1).matmul(dataset['joint3d'][i].unsqueeze(-1)).squeeze(-1)
            j3dr = j3dr[:, 1:] - j3dr[:, :1]
            v3dw = (dataset['joint3d'][i][2:] - dataset['joint3d'][i][:-2]) * 30
            contacts = torch.zeros(v3dw.shape[0], 2)
            contacts[v3dw[:, 10:12].norm(dim=2) < 0.25] = 1
            contacts = torch.cat((contacts[:1], contacts, contacts[-1:]), dim=0)
            data.append(torch.cat((accr.flatten(1), orir.flatten(1), j3dr.flatten(1)), dim=1)[1:-1])
            label.append(contacts.flatten(1)[1:-1])
        return RNNDataset(data, label, split_size=split_size, augment_fn=augment_fn, device=device)

    def NymeriaDataset(data_dir, kind, split_size=-1):
        dataset = load_nymeria_dataset(data_dir, kind)
        data, label = [], []
        for i in tqdm.trange(len(dataset['imu_acc'])):
            p = art.math.axis_angle_to_rotation_matrix(dataset['pose'][i]).view(-1, 24, 3, 3)
            j3dr = (dataset['joint3d'][i][:, 1:] - dataset['joint3d'][i][:, :1]).bmm(p[:, 0])
            accw = dataset['imu_acc'][i]
            oriw = dataset['imu_ori'][i]
            Rrw = p[:, 0].transpose(1, 2)
            accr = Rrw.unsqueeze(1).matmul(accw.unsqueeze(-1))
            orir = Rrw.unsqueeze(1).matmul(oriw)
            v3dw = (dataset['joint3d'][i][2:] - dataset['joint3d'][i][:-2]) * 30
            contacts = torch.zeros(v3dw.shape[0], 2)
            # contacts[v3dw[:, 10:12].norm(dim=2) < 0.3] = 0.5  # todo: is this useful?
            contacts[v3dw[:, 10:12].norm(dim=2) < 0.25] = 1
            contacts = torch.cat((contacts[:1], contacts, contacts[-1:]), dim=0)
            # if contacts.sum() / (v3dw.shape[0] * 2) < 0.95:
            #     visualize_contact_foot(dataset['joint3d'][i], contacts)
            data.append(torch.cat((accr.flatten(1), orir.flatten(1), j3dr.flatten(1)), dim=1)[1:-1])
            label.append(contacts.flatten(1)[1:-1])
        return RNNDataset(data, label, split_size=split_size, augment_fn=augment_fn, device=device)

    print_yellow('=================== Training RNN8 [AIST + Nymeria] ===================')

    train_dataloader = DataLoader(
        ConcatDataset([
            AISTDataset(paths.aist_dir, kind='train', split_size=200),
            NymeriaDataset(NYMERIA_DIR, kind='train', split_size=200)
        ]),
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=RNNDataset.make_collate_fn(device)
    )
    valid_dataloader = DataLoader(
        ConcatDataset([
            AISTDataset(paths.aist_dir, kind='val', split_size=200),
            NymeriaDataset(NYMERIA_DIR, kind='val', split_size=200)
        ]),
        batch_size=64,
        num_workers=2,
        pin_memory=True,
        collate_fn=RNNDataset.make_collate_fn(device)
    )

    all_labels = torch.cat(train_dataloader.dataset.datasets[0].label + train_dataloader.dataset.datasets[1].label)
    pos_weight = (1 - all_labels).sum(dim=0) / all_labels.sum(dim=0)
    rnn_bce_loss_fn = RNNLossWrapper(torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device)))

    save_dir = os.path.join(paths.weight_dir, Net.name, 'rnn8')
    net = Net().rnn8.to(device)

    train(net, train_dataloader, valid_dataloader, save_dir, loss_fn=rnn_bce_loss_fn, eval_fn=rnn_bce_loss_fn,
          num_epoch=80, num_iter_between_vald=20, clip_grad_norm=1, load_last_states=True,
          wandb_project_name='sig_mp_nymeria_aist',
          wandb_config=None, wandb_watch=True, wandb_name='rnn8', lr_scheduler_patience=10)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train RobustCap (sig_mp) on Nymeria')
    parser.add_argument('--stage', type=int, default=2, choices=[2, 3, 4, 6, 7, 8], help='RNN stage to train (2, 3, 4, 6, 7, 8)')
    args = parser.parse_args()

    if args.stage == 2:
        train_rnn2()
    elif args.stage == 3:
        train_rnn3()
    elif args.stage == 4:
        train_rnn4()
    elif args.stage == 6:
        train_rnn6()
    elif args.stage == 7:
        train_rnn7()
    elif args.stage == 8:
        train_rnn8()
    else:
        print(f"Unknown stage {args.stage}")
