r"""
RobustCap Per-RNN Diagnostic Script

Systematically tests each RNN module (2, 3, 4, 6, 7, 8) in isolation
to identify which module(s) cause poor evaluation metrics on Nymeria data.

Three testing modes per RNN:
  A) GT Input Test: Feed ground-truth inputs, measure output error
     → If error is high, the RNN itself is not learning correctly.
  B) Cascaded Test: Feed outputs from upstream RNNs (which were fed GT inputs)
     → Shows error accumulation through the pipeline.
  C) Comparison: Show training-time metric vs inference-time metric side by side.

Usage:
    /home/minghao/miniconda3/envs/robustcap/bin/python diagnose_rnn.py [--seq_idx N] [--max_seqs N]
"""
import os
import sys
sys.path.insert(0, os.getcwd())

import torch
import articulate as art
from articulate.utils.torch import *
from config import *
from torch.nn.functional import relu
import tqdm
import numpy as np

# ============================================================
# Setup
# ============================================================
NYMERIA_DIR = '/home/minghao/src/robotflow/RoHM/third_party/RobustCap/out/Nymeria_test_smplx_preprocessed.robustcap'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
body_model = art.ParametricModel(paths.smpl_file, device=device)
body_model_cpu = art.ParametricModel(paths.smpl_file)
mp_mask_t = torch.tensor(mp_mask)


def get_bbox_scale(uv):
    u_max, u_min = uv[..., 0].max(dim=-1).values, uv[..., 0].min(dim=-1).values
    v_max, v_min = uv[..., 1].max(dim=-1).values, uv[..., 1].min(dim=-1).values
    return torch.max(u_max - u_min, v_max - v_min)


def load_net():
    """Load trained Net with all RNN weights."""
    from net.sig_mp_aist_nymeria import Net
    net = Net().to(device)
    weights_path = os.path.join(paths.weight_dir, Net.name, 'best_weights.pt')
    if os.path.exists(weights_path):
        net.load_state_dict(torch.load(weights_path, weights_only=False, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"WARNING: Weights not found at {weights_path}. Using random weights.")
    net.eval()
    return net


def rnn_forward_sequence(rnn, data_seq):
    """
    Run a single RNN module on a full sequence.
    Uses the `forward` method which handles padding internally.

    Args:
        rnn: An RNN or RNNWithInit module
        data_seq: Tensor (T, input_size) or tuple for RNNWithInit

    Returns:
        Tensor (T, output_size)
    """
    with torch.no_grad():
        if isinstance(rnn, RNNWithInit):
            # RNNWithInit expects list of (data, init_label) tuples
            result = rnn([(data_seq, data_seq)])  # not correct
            # Actually for RNNWithInit.forward: x, x_init = list(zip(*x))
            # x_init should be the first label (output[0])
            # But we don't have the label. Let's just use frame-by-frame mode.
            return rnn_forward_sequence_framewise(rnn, data_seq)
        else:
            result = rnn([data_seq])
            return result[0]


def rnn_forward_sequence_framewise(rnn, data_seq):
    """
    Run RNN frame-by-frame, maintaining hidden state.
    This exactly replicates forward_online behavior.
    """
    hidden = None
    outputs = []
    with torch.no_grad():
        for t in range(data_seq.shape[0]):
            x = data_seq[t:t+1]  # (1, input_size)
            x = relu(rnn.linear1(x), inplace=False).unsqueeze(1)  # (1, 1, hidden)
            x, hidden = rnn.rnn(x, hidden)
            x = rnn.linear2(x.squeeze(1))  # (1, output_size)
            outputs.append(x.squeeze(0))
    return torch.stack(outputs)


def rnn2_forward_with_init(rnn2, data_seq, init_label):
    """
    Run RNN2 (RNNWithInit) frame-by-frame.
    For the first frame, initialize hidden state from init_label.
    """
    nd, nh = rnn2.rnn.num_layers, rnn2.rnn.hidden_size
    h_init, c_init = rnn2.init_net(init_label).view(-1, 2, nd, nh).permute(1, 2, 0, 3)
    hidden = (h_init.contiguous(), c_init.contiguous())

    outputs = []
    with torch.no_grad():
        for t in range(data_seq.shape[0]):
            x = data_seq[t:t+1]
            x = relu(rnn2.linear1(x), inplace=False).unsqueeze(1)
            x, hidden = rnn2.rnn(x, hidden)
            x = rnn2.linear2(x.squeeze(1))
            outputs.append(x.squeeze(0))
    return torch.stack(outputs)


# ============================================================
# GT Data Preparation (mirrors training NymeriaDataset)
# ============================================================

def prepare_gt_root_frame(dataset, seq_idx):
    """
    Prepare GT data in root frame (for RNN2, 3, 7, 8).
    Mirrors NymeriaDataset in train_rnn2/3/7/8.
    """
    p = art.math.axis_angle_to_rotation_matrix(dataset['pose'][seq_idx]).view(-1, 24, 3, 3).to(device)
    joint3d = dataset['joint3d'][seq_idx].to(device)

    # Root-relative joints in root frame
    j3dr = (joint3d[:, 1:] - joint3d[:, :1]).bmm(p[:, 0])  # (T, 23, 3)

    accw = dataset['imu_acc'][seq_idx].to(device)
    oriw = dataset['imu_ori'][seq_idx].to(device)

    Rrw = p[:, 0].transpose(1, 2)  # World-to-Root rotation
    accr = Rrw.unsqueeze(1).matmul(accw.unsqueeze(-1)).squeeze(-1)  # (T, 6, 3)
    orir = Rrw.unsqueeze(1).matmul(oriw)  # (T, 6, 3, 3)

    # Root velocity (for RNN3 label)
    # Training code: v3dw = (dataset['joint3d'][i][2:] - dataset['joint3d'][i][:-2]) * 30
    # Then takes [:, 0] for root joint, pads with zeros at start and end
    v3dw_all_joints = (joint3d[2:] - joint3d[:-2]) * 30  # (T-2, 24, 3)
    v3dw_root = v3dw_all_joints[:, 0]  # (T-2, 3)
    v3dw_full = torch.cat((torch.zeros(1, 3, device=device), v3dw_root, torch.zeros(1, 3, device=device)), dim=0) / vel_scale
    v3dr = Rrw.matmul(v3dw_full.unsqueeze(-1)).squeeze(-1)  # (T, 3)

    # Contacts (for RNN8 label)
    v3dw_all = v3dw_all_joints  # (T-2, 24, 3)
    contacts = torch.zeros(v3dw_all.shape[0], 2, device=device)
    contacts[v3dw_all[:, 10:12].norm(dim=2) < 0.25] = 1
    contacts = torch.cat((contacts[:1], contacts, contacts[-1:]), dim=0)  # (T, 2)

    # Pose label (for RNN7)
    p_for_label = art.math.axis_angle_to_rotation_matrix(dataset['pose'][seq_idx]).view(-1, 24, 3, 3).to(device)
    p_for_label[:, 0] = torch.eye(3, device=device)
    glbp = body_model.forward_kinematics_R(p_for_label)
    p6d = art.math.rotation_matrix_to_r6d(glbp).view(-1, 24 * 6)  # (T, 144)

    # Special orientation for RNN7: first 5 IMUs in root frame, 6th stays world frame
    orir_rnn7 = oriw.clone()
    orir_rnn7[:, :5] = Rrw.unsqueeze(1).matmul(oriw[:, :5])

    return {
        'accr': accr,           # (T, 6, 3)
        'orir': orir,           # (T, 6, 3, 3)
        'orir_rnn7': orir_rnn7, # (T, 6, 3, 3) - special for RNN7
        'j3dr': j3dr,           # (T, 23, 3)
        'v3dr': v3dr,           # (T, 3)
        'contacts': contacts,   # (T, 2)
        'p6d': p6d,             # (T, 144)
        'Rrw': Rrw,             # (T, 3, 3)
        'p': p,                 # (T, 24, 3, 3)
    }


def prepare_gt_camera_frame(dataset, seq_idx):
    """
    Prepare GT data in camera frame (for RNN4, 6).
    Mirrors NymeriaDataset __getitem__ in train_rnn4/6.
    Uses a FIXED virtual camera (same as evaluate_nymeria.py).
    """
    accw = dataset['imu_acc'][seq_idx].to(device)  # (T, 6, 3)
    oriw = dataset['imu_ori'][seq_idx].to(device)  # (T, 6, 3, 3)
    joint3d = dataset['joint3d'][seq_idx].to(device)  # (T, 24, 3)
    sync_3d_mp = dataset['sync_3d_mp'][seq_idx].to(device)  # (T, 33, 3)

    # Center to first frame root
    root = joint3d[0, 0].clone()
    j3dw = joint3d - root  # (T, 24, 3)
    j3dw_mp = sync_3d_mp - root  # (T, 33, 3)

    # Map SMPL joints to MediaPipe positions (same as training)
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

    # Fixed virtual camera (canonical view, same as eval)
    Rwc0 = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1.]], device=device)
    Rc0c = art.math.generate_random_rotation_matrix_constrained(n=1, y=(0, 0), p=(0, 0), r=(0, 0))[0].to(device)
    Rcw = Rwc0.mm(Rc0c).t()

    # Transform to camera frame
    accc = Rcw.matmul(accw.reshape(-1, 6, 3, 1)).squeeze(-1)  # (T, 6, 3)
    oric = Rcw.matmul(oriw.reshape(-1, 6, 3, 3))  # (T, 6, 3, 3)
    j3dc = (Rcw.matmul(j3dw.unsqueeze(-1))).squeeze(-1)  # (T, 24, 3)
    j3dc_mp = (Rcw.matmul(j3dw_mp.unsqueeze(-1))).squeeze(-1)  # (T, 33, 3)

    # Add translation (fixed: [0, 0, 5.5])
    random_tranc = torch.tensor([0., 0., 5.5], device=device)
    random_tranc[2] -= j3dc[..., -1].min()
    j3dc = j3dc + random_tranc
    j3dc_mp = j3dc_mp + random_tranc

    # 2D projection (normalized rays)
    j2dc = j3dc_mp / j3dc_mp[..., -1:]  # (T, 33, 3)

    # Confidence = 1.0 (no noise for deterministic eval)
    j2dc[..., -1] = 1.0

    # BBox normalization
    bbox_scale = get_bbox_scale(j2dc).view(-1, 1, 1)
    j2dc[..., :2] = j2dc[..., :2] / bbox_scale

    # Root-relative subtraction
    j2dc[:, 24:, :2] = j2dc[:, 24:, :2] - j2dc[:, 23:24, :2]
    j2dc[:, :23, :2] = j2dc[:, :23, :2] - j2dc[:, 23:24, :2]

    # Labels
    j3dc_root_rel = j3dc[:, 1:] - j3dc[:, :1]  # (T, 23, 3) - RNN4 label
    tranc = j3dc[:, 0]  # (T, 3) - RNN6 label (root position in camera frame)

    return {
        'accc': accc,                # (T, 6, 3)
        'oric': oric,                # (T, 6, 3, 3)
        'j2dc': j2dc,               # (T, 33, 3) - processed 2D keypoints
        'j3dc_root_rel': j3dc_root_rel,  # (T, 23, 3) - RNN4 GT label
        'tranc': tranc,              # (T, 3) - RNN6 GT label
        'Rcw': Rcw,                  # (3, 3)
        'random_tranc': random_tranc,# (3,)
    }


# ============================================================
# Per-RNN Tests
# ============================================================

def test_rnn2(net, gt_root):
    """Test RNN2 with GT inputs. Returns predicted j3dr and error."""
    accr = gt_root['accr']  # (T, 6, 3)
    orir = gt_root['orir']  # (T, 6, 3, 3)
    j3dr_gt = gt_root['j3dr']  # (T, 23, 3)

    # Build input: cat(accr, orir) → (T, 72)
    data = torch.cat((accr.flatten(1), orir.flatten(1)), dim=1)  # (T, 72)

    # Run RNN2 frame-by-frame (like forward_online)
    pred = rnn_forward_sequence_framewise(net.rnn2, data)  # (T, 69)

    # Also test with RNNWithInit (give GT first-frame label as init)
    pred_with_init = rnn2_forward_with_init(net.rnn2, data, j3dr_gt[0].flatten())

    # Compute error
    gt_flat = j3dr_gt.flatten(1)  # (T, 69)
    err_no_init = (pred - gt_flat).view(-1, 23, 3).norm(dim=2).mean().item()
    err_with_init = (pred_with_init - gt_flat).view(-1, 23, 3).norm(dim=2).mean().item()

    return pred, pred_with_init, err_no_init, err_with_init


def test_rnn3(net, gt_root, j3dr_input=None, input_label='GT'):
    """Test RNN3 with GT or predicted j3dr."""
    accr = gt_root['accr']
    orir = gt_root['orir']
    v3dr_gt = gt_root['v3dr']
    j3dr = gt_root['j3dr'] if j3dr_input is None else j3dr_input.view(-1, 23, 3)

    data = torch.cat((accr.flatten(1), orir.flatten(1), j3dr.flatten(1)), dim=1)

    pred = rnn_forward_sequence_framewise(net.rnn3, data)
    gt_flat = v3dr_gt.flatten(1)

    # Velocity error
    err = (pred - gt_flat).view(-1, 3).norm(dim=1).mean().item()

    # Also compute integrated position error over 1 second (30 frames)
    # v * vel_scale / 60 gives per-frame displacement
    pred_disp = pred * vel_scale / 60  # (T, 3)
    gt_disp = gt_flat * vel_scale / 60

    pred_pos = torch.cumsum(pred_disp, dim=0)
    gt_pos = torch.cumsum(gt_disp, dim=0)
    pos_err = (pred_pos - gt_pos).norm(dim=1).mean().item()

    return pred, err, pos_err


def test_rnn4(net, gt_cam):
    """Test RNN4 with GT camera-frame inputs."""
    accc = gt_cam['accc']
    oric = gt_cam['oric']
    j2dc = gt_cam['j2dc']
    j3dc_gt = gt_cam['j3dc_root_rel']

    data = torch.cat((accc.flatten(1), oric.flatten(1), j2dc.flatten(1)), dim=1)

    pred = rnn_forward_sequence_framewise(net.rnn4, data)
    gt_flat = j3dc_gt.flatten(1)

    err = (pred - gt_flat).view(-1, 23, 3).norm(dim=2).mean().item()
    return pred, err


def test_rnn6(net, gt_cam, j3dc_input=None, input_label='GT'):
    """Test RNN6 with GT camera-frame inputs."""
    accc = gt_cam['accc']
    oric = gt_cam['oric']
    j2dc = gt_cam['j2dc']
    tranc_gt = gt_cam['tranc']

    if j3dc_input is None:
        j3dc = gt_cam['j3dc_root_rel']
    else:
        j3dc = j3dc_input.view(-1, 23, 3)

    data = torch.cat((accc.flatten(1), oric.flatten(1), j2dc.flatten(1), j3dc.flatten(1)), dim=1)

    pred = rnn_forward_sequence_framewise(net.rnn6, data)
    gt_flat = tranc_gt

    err = (pred - gt_flat).norm(dim=1).mean().item()
    return pred, err


def test_rnn7(net, gt_root, j3dr_input=None, input_label='GT'):
    """Test RNN7 with GT or predicted j3dr. Returns FK joint error."""
    accr = gt_root['accr']
    orir_rnn7 = gt_root['orir_rnn7']
    p6d_gt = gt_root['p6d']
    j3dr = gt_root['j3dr'] if j3dr_input is None else j3dr_input.view(-1, 23, 3)

    data = torch.cat((accr.flatten(1), orir_rnn7.flatten(1), j3dr.flatten(1)), dim=1)

    pred = rnn_forward_sequence_framewise(net.rnn7, data)

    # 6D rotation error (raw)
    err_6d = (pred - p6d_gt).pow(2).mean().sqrt().item()

    # FK joint position error (more meaningful)
    j_zero = body_model.get_zero_pose_joint_and_vertex()[0]
    b = body_model.joint_position_to_bone_vector(j_zero.unsqueeze(0)).view(24, 3, 1).to(device)

    def fk_from_6d(p6d_batch):
        p = art.math.r6d_to_rotation_matrix(p6d_batch).view(-1, 24, 3, 3)
        pb = torch.stack([p[:, body_model.parent[i]].matmul(b[i]) for i in range(1, 24)], dim=1)
        pb = torch.cat((torch.zeros(p.shape[0], 1, 3, device=device), pb.squeeze(-1)), dim=1)
        return body_model.bone_vector_to_joint_position(pb)

    with torch.no_grad():
        joints_pred = fk_from_6d(pred)
        joints_gt = fk_from_6d(p6d_gt)

    err_fk = (joints_pred - joints_gt).norm(dim=2).mean().item()

    return pred, err_6d, err_fk


def test_rnn8(net, gt_root, j3dr_input=None, input_label='GT'):
    """Test RNN8 with GT or predicted j3dr."""
    accr = gt_root['accr']
    orir = gt_root['orir']
    contacts_gt = gt_root['contacts']
    j3dr = gt_root['j3dr'] if j3dr_input is None else j3dr_input.view(-1, 23, 3)

    data = torch.cat((accr.flatten(1), orir.flatten(1), j3dr.flatten(1)), dim=1)

    pred_logits = rnn_forward_sequence_framewise(net.rnn8, data)
    pred_probs = pred_logits.sigmoid()

    # BCE Loss
    bce = torch.nn.BCEWithLogitsLoss()(pred_logits, contacts_gt).item()

    # Accuracy
    pred_binary = (pred_probs > 0.5).float()
    accuracy = (pred_binary == contacts_gt).float().mean().item()

    return pred_logits, bce, accuracy


def test_full_pipeline_against_forward_online(net, dataset, seq_idx, gt_cam):
    """
    Run forward_online on the same data to get mpjpe,
    allowing direct comparison with individual RNN tests.
    """
    from net.sig_mp_aist_nymeria import Net as NetClass

    accw = dataset['imu_acc'][seq_idx].to(device)
    oriw = dataset['imu_ori'][seq_idx].to(device)
    j3dw = dataset['joint3d'][seq_idx].to(device)
    sync_3d_mp = dataset['sync_3d_mp'][seq_idx].to(device)

    Rcw = gt_cam['Rcw']
    random_tranc = gt_cam['random_tranc']

    # Prepare forward_online inputs (same as evaluate_nymeria.py)
    root_offset = j3dw[0, 0].clone()
    j3dw_centered = j3dw - root_offset
    j3dw_mp_centered = sync_3d_mp - root_offset

    # Map MP joints
    j3dw_mp_centered[:, 11] = j3dw_centered[:, 16].clone()
    j3dw_mp_centered[:, 12] = j3dw_centered[:, 17].clone()
    j3dw_mp_centered[:, 13] = j3dw_centered[:, 18].clone()
    j3dw_mp_centered[:, 14] = j3dw_centered[:, 19].clone()
    j3dw_mp_centered[:, 15] = j3dw_centered[:, 20].clone()
    j3dw_mp_centered[:, 16] = j3dw_centered[:, 21].clone()
    j3dw_mp_centered[:, 23] = j3dw_centered[:, 1].clone()
    j3dw_mp_centered[:, 24] = j3dw_centered[:, 2].clone()
    j3dw_mp_centered[:, 25] = j3dw_centered[:, 4].clone()
    j3dw_mp_centered[:, 26] = j3dw_centered[:, 5].clone()
    j3dw_mp_centered[:, 27] = j3dw_centered[:, 7].clone()
    j3dw_mp_centered[:, 28] = j3dw_centered[:, 8].clone()

    accc = Rcw.matmul(accw.unsqueeze(-1)).squeeze(-1)
    oric = Rcw.matmul(oriw)
    j3dc_mp = Rcw.matmul(j3dw_mp_centered.unsqueeze(-1)).squeeze(-1)
    j3dc = Rcw.matmul(j3dw_centered.unsqueeze(-1)).squeeze(-1)

    j3dc = j3dc + random_tranc
    j3dc_mp = j3dc_mp + random_tranc

    j2dc = j3dc_mp / j3dc_mp[..., -1:]
    j2dc[..., -1] = 1.0
    bbox_scale = get_bbox_scale(j2dc).view(-1, 1, 1)
    j2dc[..., :2] = j2dc[..., :2] / bbox_scale
    j2dc[:, 24:, :2] = j2dc[:, 24:, :2] - j2dc[:, 23:24, :2]
    j2dc[:, :23, :2] = j2dc[:, :23, :2] - j2dc[:, 23:24, :2]

    # Set gravity
    g_world = torch.tensor([0., -1., 0.], device=device).reshape(3, 1)
    g_cam = Rcw.matmul(g_world).view(3)
    NetClass.gravityc = g_cam

    first_tran = j3dc[0, 0]
    net.reset_states()
    pose_preds = []
    tran_preds = []
    for i in range(len(j2dc)):
        if i == 0:
            p, t = net.forward_online(j2dc[i], accc[i], oric[i], first_tran)
        else:
            p, t = net.forward_online(j2dc[i], accc[i], oric[i])
        pose_preds.append(p)
        tran_preds.append(t)

    pose_pred = torch.stack(pose_preds)
    tran_pred = torch.stack(tran_preds)

    # Compute MPJPE against GT (chunk to avoid OOM)
    gt_pose = art.math.axis_angle_to_rotation_matrix(dataset['pose'][seq_idx]).view(-1, 24, 3, 3).to(device)
    gt_pose[:, 0] = Rcw.matmul(gt_pose[:, 0])  # align root to camera frame

    T = gt_pose.shape[0]
    chunk_size = 500
    J_regressor = torch.from_numpy(np.load(paths.j_regressor_dir)).float().to(device)

    mpjpe_sum, pve_sum, count = 0., 0., 0
    for ci in range(0, T, chunk_size):
        ce = min(ci + chunk_size, T)
        with torch.no_grad():
            _, _, gt_v = body_model.forward_kinematics(gt_pose[ci:ce], calc_mesh=True)
            _, _, pr_v = body_model.forward_kinematics(pose_pred[ci:ce].to(device), calc_mesh=True)

            J_batch = J_regressor[None, :].expand(gt_v.shape[0], -1, -1)
            gt_kp = torch.matmul(J_batch, gt_v)[:, :14]
            pr_kp = torch.matmul(J_batch, pr_v)[:, :14]
            gt_kp = gt_kp - gt_kp[:, [0], :]
            pr_kp = pr_kp - pr_kp[:, [0], :]

            mpjpe_sum += (gt_kp - pr_kp).norm(dim=2).mean().item() * (ce - ci)
            pve_sum += (gt_v - pr_v).norm(dim=2).mean().item() * (ce - ci)
            count += (ce - ci)

            del gt_v, pr_v, gt_kp, pr_kp, J_batch
            torch.cuda.empty_cache()

    mpjpe = mpjpe_sum / count
    pve = pve_sum / count

    return mpjpe, pve


# ============================================================
# Main Diagnostic
# ============================================================

def run_diagnostics(seq_idx=0, max_seqs=1, max_frames=2000):
    # Load data
    dataset_path = os.path.join(NYMERIA_DIR, 'test_nymeria.pt')
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(NYMERIA_DIR, 'val_nymeria.pt')
    print(f"Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path, weights_only=False)

    net = load_net()

    n_seqs = min(len(dataset['pose']), max_seqs)
    print(f"\nRunning diagnostics on {n_seqs} sequence(s) starting from index {seq_idx}...")
    print(f"{'='*80}")

    for s in range(seq_idx, min(seq_idx + n_seqs, len(dataset['pose']))):
        T_full = dataset['pose'][s].shape[0]
        T = min(T_full, max_frames)
        if T < T_full:
            # Truncate all sequence data to max_frames
            for key in dataset:
                if isinstance(dataset[key], list) and len(dataset[key]) > s:
                    if hasattr(dataset[key][s], 'shape') and dataset[key][s].shape[0] == T_full:
                        dataset[key][s] = dataset[key][s][:T]
        print(f"\n{'='*80}")
        print(f"Sequence {s}: {T_full} total frames, using {T} frames")
        print(f"{'='*80}")

        # Prepare GT data
        gt_root = prepare_gt_root_frame(dataset, s)
        gt_cam = prepare_gt_camera_frame(dataset, s)

        # ========================
        # A. Individual RNN Tests (GT Inputs)
        # ========================
        print(f"\n--- A. Individual RNN Tests (Ground Truth Inputs) ---")

        # RNN2
        rnn2_pred, rnn2_pred_init, rnn2_err, rnn2_err_init = test_rnn2(net, gt_root)
        print(f"  RNN2 (Inertial Pose):     Joint Pos Error = {rnn2_err:.4f} m  (no init)")
        print(f"  RNN2 (Inertial Pose):     Joint Pos Error = {rnn2_err_init:.4f} m  (with GT init)")

        # RNN3
        _, rnn3_err, rnn3_pos_err = test_rnn3(net, gt_root, input_label='GT')
        print(f"  RNN3 (Root Velocity):     Velocity Error = {rnn3_err:.4f}    Integrated Pos Error = {rnn3_pos_err:.4f} m")

        # RNN4
        rnn4_pred, rnn4_err = test_rnn4(net, gt_cam)
        print(f"  RNN4 (Visual Pose):       Joint Pos Error = {rnn4_err:.4f} m")

        # RNN6
        _, rnn6_err = test_rnn6(net, gt_cam, input_label='GT')
        print(f"  RNN6 (Global Trans):      Translation Error = {rnn6_err:.4f} m")

        # RNN7
        _, rnn7_err_6d, rnn7_err_fk = test_rnn7(net, gt_root, input_label='GT')
        print(f"  RNN7 (Rotation/IK):       6D RMSE = {rnn7_err_6d:.4f}    FK Joint Error = {rnn7_err_fk:.4f} m")

        # RNN8
        _, rnn8_bce, rnn8_acc = test_rnn8(net, gt_root, input_label='GT')
        print(f"  RNN8 (Contact):           BCE = {rnn8_bce:.4f}    Accuracy = {rnn8_acc:.4f}")

        # ========================
        # B. Cascaded Tests (Error Propagation)
        # ========================
        print(f"\n--- B. Cascaded Tests (Upstream RNN outputs as input) ---")

        # RNN2 → RNN3
        _, rnn3_err_cascade, rnn3_pos_err_cascade = test_rnn3(net, gt_root, j3dr_input=rnn2_pred_init, input_label='RNN2 pred')
        print(f"  RNN2→RNN3 (Cascade):      Velocity Error = {rnn3_err_cascade:.4f}    Integrated Pos Error = {rnn3_pos_err_cascade:.4f} m")

        # RNN2 → RNN7
        _, rnn7_err_6d_cascade, rnn7_err_fk_cascade = test_rnn7(net, gt_root, j3dr_input=rnn2_pred_init, input_label='RNN2 pred')
        print(f"  RNN2→RNN7 (Cascade):      6D RMSE = {rnn7_err_6d_cascade:.4f}    FK Joint Error = {rnn7_err_fk_cascade:.4f} m")

        # RNN2 → RNN8
        _, rnn8_bce_cascade, rnn8_acc_cascade = test_rnn8(net, gt_root, j3dr_input=rnn2_pred_init, input_label='RNN2 pred')
        print(f"  RNN2→RNN8 (Cascade):      BCE = {rnn8_bce_cascade:.4f}    Accuracy = {rnn8_acc_cascade:.4f}")

        # RNN4 → RNN6
        _, rnn6_err_cascade = test_rnn6(net, gt_cam, j3dc_input=rnn4_pred, input_label='RNN4 pred')
        print(f"  RNN4→RNN6 (Cascade):      Translation Error = {rnn6_err_cascade:.4f} m")

        # ========================
        # C. Full Forward Online Pipeline
        # ========================
        print(f"\n--- C. Full forward_online Pipeline ---")
        mpjpe, pve = test_full_pipeline_against_forward_online(net, dataset, s, gt_cam)
        print(f"  forward_online:            MPJPE = {mpjpe:.4f} m    PVE = {pve:.4f} m")

        # ========================
        # D. Data Shape Verification
        # ========================
        print(f"\n--- D. Data Shape Verification ---")
        print(f"  joint3d shape:     {dataset['joint3d'][s].shape}")
        print(f"  sync_3d_mp shape:  {dataset['sync_3d_mp'][s].shape}")
        print(f"  imu_acc shape:     {dataset['imu_acc'][s].shape}")
        print(f"  imu_ori shape:     {dataset['imu_ori'][s].shape}")
        print(f"  pose shape:        {dataset['pose'][s].shape}")

        # Check value ranges
        j3d = dataset['joint3d'][s]
        print(f"  joint3d value range: [{j3d.min().item():.3f}, {j3d.max().item():.3f}]")
        print(f"  joint3d root range:  [{j3d[:, 0].min().item():.3f}, {j3d[:, 0].max().item():.3f}]")
        acc = dataset['imu_acc'][s]
        print(f"  imu_acc value range: [{acc.min().item():.3f}, {acc.max().item():.3f}]")

        # ========================
        # E. Summary Table
        # ========================
        print(f"\n{'='*80}")
        print(f"SUMMARY for Sequence {s}")
        print(f"{'='*80}")
        print(f"{'Module':<20} {'GT Input Error':<25} {'Cascaded Error':<25} {'Degradation'}")
        print(f"{'-'*90}")
        print(f"{'RNN2 (j3dr)':<20} {rnn2_err_init:<25.4f} {'N/A (first stage)':<25}")
        print(f"{'RNN3 (velocity)':<20} {rnn3_err:<25.4f} {rnn3_err_cascade:<25.4f} {(rnn3_err_cascade/rnn3_err - 1)*100:+.1f}%")
        print(f"{'RNN4 (visual j3d)':<20} {rnn4_err:<25.4f} {'N/A (first stage)':<25}")
        print(f"{'RNN6 (translation)':<20} {rnn6_err:<25.4f} {rnn6_err_cascade:<25.4f} {(rnn6_err_cascade/rnn6_err - 1)*100:+.1f}%")
        rnn7_label = 'FK joint err (m)'
        print(f"{'RNN7 (rotation)':<20} {rnn7_err_fk:<25.4f} {rnn7_err_fk_cascade:<25.4f} {(rnn7_err_fk_cascade/rnn7_err_fk - 1)*100:+.1f}%")
        print(f"{'RNN8 (contact)':<20} {rnn8_acc:<25.4f} {rnn8_acc_cascade:<25.4f} {(rnn8_acc_cascade/rnn8_acc - 1)*100:+.1f}%")
        print(f"{'forward_online':<20} MPJPE={mpjpe:.4f} m, PVE={pve:.4f} m")
        print(f"{'='*80}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RobustCap Per-RNN Diagnostic')
    parser.add_argument('--seq_idx', type=int, default=0, help='Starting sequence index')
    parser.add_argument('--max_seqs', type=int, default=1, help='Maximum sequences to test')
    parser.add_argument('--max_frames', type=int, default=2000, help='Maximum frames per sequence')
    args = parser.parse_args()

    run_diagnostics(seq_idx=args.seq_idx, max_seqs=args.max_seqs, max_frames=args.max_frames)
