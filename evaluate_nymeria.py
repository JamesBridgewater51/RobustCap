import torch
import os
import tqdm
import articulate as art
import numpy as np
import cv2
import random
from config import paths
from net.sig_mp_aist_nymeria import Net, NYMERIA_DIR
import ipdb
import signal
import traceback as tb
import sys

def ipdb_safety_net():
    """Attaches a "safety net" for unexpected errors in a Python script.

    When called, PDB will be automatically opened when either (a) the user hits Ctrl+C
    or (b) we encounter an uncaught exception. Helpful for bypassing minor errors,
    diagnosing problems, and rescuing unsaved models.
    """

    # Open PDB on Ctrl+C
    def handler(sig, frame):
        ipdb.set_trace()

    signal.signal(signal.SIGINT, handler)

    # Open PDB when we encounter an uncaught exception
    def excepthook(type_, value, traceback):  # pragma: no cover (impossible to test)
        tb.print_exception(type_, value, traceback, limit=100)
        ipdb.post_mortem(traceback)

    sys.excepthook = excepthook


# Import helper functions directly from sig_mp_nymeria to ensure consistency if possible,
# but get_bbox_scale is a global function there.
# To avoid robust import issues with relative paths, we redefine it here identically.
def get_bbox_scale(uv):
    u_max, u_min = uv[..., 0].max(dim=-1).values, uv[..., 0].min(dim=-1).values
    v_max, v_min = uv[..., 1].max(dim=-1).values, uv[..., 1].min(dim=-1).values
    return torch.max(u_max - u_min, v_max - v_min)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
body_model = art.ParametricModel(paths.smpl_file, device=device)

def get_virtual_camera_transform(fixed_view=True):
    r"""
    Replicates the synthetic camera logic from training (sig_mp_nymeria.py).
    Training Logic:
        Rwc0 = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1.]])
        Rc0c = art.math.generate_random_rotation_matrix_constrained(...)
        Rcw = Rwc0.mm(Rc0c).t()
    """
    Rwc0 = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1.]], device=device)
    
    if fixed_view:
        # User requested CANONICAL view for evaluation/vis.
        # Generating with 0 params yields Identity rotation (or close to).
        # generate_random_rotation_matrix_constrained(n=1, y=(0,0), p=(0,0), r=(0,0))
        Rc0c = art.math.generate_random_rotation_matrix_constrained(n=1, y=(0, 0), p=(0, 0), r=(0, 0))[0].to(device)
    else:
        # Replicate training randomness if desired
        Rc0c = art.math.generate_random_rotation_matrix_constrained(n=1, y=(-90, 90), p=(-30, 30), r=(-5, 5))[0].to(device)
        
    Rcw = Rwc0.mm(Rc0c).t()
    return Rcw

def get_virtual_translation(j3dc, fixed_view=True):
    r"""
    Replicates training translation logic:
        random_tranc = lerp([-1,-1,3], [1,1,8])
        random_tranc[2] -= j3dc[..., -1].min()
        j3dc = j3dc + random_tranc
    """
    if fixed_view:
        # Use Mean of range: [-1,1]->0, [3,8]->5.5
        random_tranc = torch.tensor([0., 0., 5.5], device=device)
    else:
        random_tranc = art.math.lerp(torch.tensor([-1, -1, 3.]), torch.tensor([1, 1, 8.]), torch.rand(3)).to(device)
    
    # Adhere to training logic: Ensure depth is relative to min depth
    # "random_tranc[2] -= j3dc[..., -1].min()"
    # This pushes the character so that its closest point is at random_tranc[2] depth (e.g. 3m-8m away).
    min_z = j3dc[..., -1].min()
    random_tranc[2] = random_tranc[2] - min_z
    return random_tranc

def view_nymeria_ours(seq_idx=0, vis=True, run_smplify=True, save_mp4=False):
    dataset_path = os.path.join(NYMERIA_DIR, 'test_nymeria.pt')
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(NYMERIA_DIR, 'val_nymeria.pt')
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    print(f"Loading data from {dataset_path}...")
    dataset = torch.load(dataset_path, weights_only=False)
    
    if seq_idx >= len(dataset['pose']):
        print(f"Sequence index {seq_idx} out of range (max {len(dataset['pose'])-1})")
        return

    # Extract & Prepare Data (Strictly following Training Logic)
    # Tensors must be moved to device for operations
    accw = dataset['imu_acc'][seq_idx].reshape(-1, 6, 3, 1).to(device) # (T, 6, 3, 1)
    oriw = dataset['imu_ori'][seq_idx].reshape(-1, 6, 3, 3).to(device) # (T, 6, 3, 3)
    j3dw_mp = dataset['sync_3d_mp'][seq_idx].reshape(-1, 33, 3, 1).to(device)
    j3dw = dataset['joint3d'][seq_idx].reshape(-1, 24, 3, 1).to(device)

    # [Correctness] Center the sequence
    root_offset = j3dw[0, 0].clone()
    j3dw = j3dw - root_offset
    j3dw_mp = j3dw_mp - root_offset
    
    # 1. Coordinate Transform (World -> Virtual Camera)
    Rcw = get_virtual_camera_transform(fixed_view=True) # (3, 3)
    
    accc = Rcw.matmul(accw) # (T, 6, 3, 1)
    oric = Rcw.matmul(oriw) # (T, 6, 3, 3)
    j3dc = Rcw.matmul(j3dw).squeeze(-1)       # (T, 24, 3)
    j3dc_mp = Rcw.matmul(j3dw_mp).squeeze(-1) # (T, 33, 3)
    
    # 2. Translation & Depth Adjustment
    random_tranc = get_virtual_translation(j3dc, fixed_view=True)
    
    # Training applies translation to BOTH
    j3dc = j3dc + random_tranc
    j3dc_mp = j3dc_mp + random_tranc
    
    # 3. Projection & Normalization (Normalized Rays)
    # j2dc = j3dc_mp / j3dc_mp[..., -1:]
    j2dc = j3dc_mp / j3dc_mp[..., -1:]
    
    # 4. Confidence Injection
    # Training samples from 'syn_c.pt'. For eval, we want high confidence (1.0) usually,
    # or mean confidence. Code uses 1.0 (mean of synthetic might be lower).
    # "p = self.conf[rand]" -> "j2dc[..., -1:] = p"
    # Let's use p=1.0 for clear visibility/eval.
    p = 1.0
    
    # Training: j2dc[..., :2] = torch.normal(..., 0.003 * (1-p)) # Noise
    # Eval: No noise injection for deterministic output.
    j2dc[..., -1] = p
    
    # 5. BBox Scaling (Crucial Step)
    # j2dc[..., :2] = j2dc[..., :2] / (get_bbox_scale(j2dc)).view(-1, 1, 1)
    bbox_scale = get_bbox_scale(j2dc).view(-1, 1, 1)
    j2dc[..., :2] = j2dc[..., :2] / bbox_scale

    # 6. Root-Relative Subtraction (Crucial Step order)
    # j2dc[:, 24:, :2] = j2dc[:, 24:, :2] - j2dc[:, 23:24, :2]
    # j2dc[:, :23, :2] = j2dc[:, :23, :2] - j2dc[:, 23:24, :2]
    j2dc[:, 24:, :2] = j2dc[:, 24:, :2] - j2dc[:, 23:24, :2]
    j2dc[:, :23, :2] = j2dc[:, :23, :2] - j2dc[:, 23:24, :2]
    
    # Now `j2dc` is ready for the Network.
    j2dc_input = j2dc
    
    # Input flattening for Net
    accc_input = accc.squeeze(-1) # (T, 6, 3)
    oric_input = oric # (T, 6, 3, 3)
    
    g_world = torch.tensor([0., -1., 0.], device=device).reshape(3, 1)
    g_cam = Rcw.matmul(g_world).view(3)
    Net.gravityc = g_cam
    
    net = Net().to(device)
    weights_path = os.path.join(paths.weight_dir, Net.name, 'best_weights.pt')
    if os.path.exists(weights_path):
        net.load_state_dict(torch.load(weights_path, weights_only=False))
    else:
        print(f"Warning: Weights not found at {weights_path}")
    net.eval()
    
    pose_pred, tran_pred = [], []
    # Note: first_tran should be ground truth start for initialization?
    # In view_aist: first_tran = tranc[0]
    # Here tranc is `random_tranc` + dynamic motion?
    # Training does: "label = j3dc.flatten(1)". j3dc includes `random_tranc`.
    # So `tran_pred` should match `j3dc[0, 0]` (Root position).
    first_tran_gt = j3dc[0, 0] # Root of first frame
    
    print("Running Online Inference...")
    for i in tqdm.trange(len(j2dc_input)):
        if i == 0:
            p, t = net.forward_online(j2dc_input[i], accc_input[i], oric_input[i], first_tran_gt)
        else:
            p, t = net.forward_online(j2dc_input[i], accc_input[i], oric_input[i]) # uses last_tran
        
        pose_pred.append(p)
        tran_pred.append(t)
        
    pose_pred = torch.stack(pose_pred) # (T, 24, 3, 3)
    tran_pred = torch.stack(tran_pred) # (T, 3)
    
    run_smplify = False
    if run_smplify:
        print("Running SMPLify Optimization...")
        # Issue: SMPLify needs PIXEL coordinates (j2dc_pixels) + Camera Intrisics (K).
        # But we only have Normalized Rays (j2dc_input) where K was "implicitly removed" by division.
        # Training does: j2dc = j3dc_mp / z. This IS normalized ray (if we assume fx=fy=1, cx=cy=0).
        # So "K" for SMPLify should be Identity?
        # Or did training assume a specific K for synthetic generation?
        # "j2dc = j3dc_mp / j3dc_mp[..., -1:]" -> x/z, y/z, 1.
        # This corresponds to K = Identity.
        # So we pass K = Identity to smplify_runner and pass the Normalized Rays as "pixels".
        # Why? Because smplify projects: K @ (R@v + t).
        # Input j2dc_opt is observed "pixels".
        # If K=I, predicted projection is x/z, y/z.
        # Error = || (x/z, y/z) - observed(x/z, y/z) ||.
        # This matches!
        
        K_identity = torch.eye(3, device=device)
        
        # However, j2dc_input used for Net was FURTHER normalized by bbox_scale and centered!
        # SMPLify likely needs the "Raw Normalized Rays" (x/z, y/z) BEFORE bbox scaling/centering.
        # Training code:
        # 1. j2dc = j3dc_mp / z (Raw Projection)
        # 2. Add noise
        # 3. Scale/Center for NET INPUT
        # We need state (1) or (2) for SMPLify.
        
        j2dc_for_opt = j3dc_mp / j3dc_mp[..., -1:] # Recompute or cache state (1)
        j2dc_for_opt[..., -1] = 1.0 # Confidence
        
        # We need to import smplify_runner
        # assuming it's available or user handles import
        from net.smplify.run import smplify_runner
        
        # oric is (T, 6, 3, 3)
        pose_pred, tran_pred, _ = smplify_runner(
            pose_pred, tran_pred, j2dc_for_opt, oric_input, 
            batch_size=pose_pred.shape[0], lr=0.001, use_lbfgs=True, opt_steps=1, cam_k=K_identity, use_head=True
        )

    if vis:
        print("Visualizing...")
        # Visualize Predicted Pose vs GT (j3dc derived from GT)
        # We need to reconstruct full body mesh from GT pose parameters?
        # Or just view predicted?
        # Evaluate.py: body_model.view_motion([pose, posec[:len]])
        # Pass GT Rotation (posec - which is `pose` in dataset)
        # Note: GT data['pose'] is axis-angle. Convert to rotmat.
        gt_pose_aa = dataset['pose'][seq_idx].to(device)
        gt_pose_rotmat = art.math.axis_angle_to_rotation_matrix(gt_pose_aa).view(-1, 24, 3, 3)
        
        # GT Global Orientation must be rotated by Rcw to match View!
        # Training does: "p[:, 0] = Rwc0.mm(Rc0c).t().matmul(p[:, 0])" -> Rcw @ p[0]
        # We did this for j3dw and imu, but `gt_pose_rotmat` is still World Frame.
        gt_pose_rotmat[:, 0] = Rcw.matmul(gt_pose_rotmat[:, 0])
        
        body_model_cpu = art.ParametricModel(paths.smpl_file, device=torch.device('cpu'))
        body_model_cpu.view_motion([pose_pred.cpu(), gt_pose_rotmat[:len(pose_pred)].cpu()])
        
import utils
import config  # pyright: ignore[reportImplicitRelativeImport]


def cal_mpjpe(pose, gt_pose, cal_pampjpe=False):
    J_regressor = torch.from_numpy(np.load(config.paths.j_regressor_dir)).float().to(torch.device('cpu'))
    body_model_cpu = art.ParametricModel(paths.smpl_file, device=torch.device('cpu'))
    _, _, gt_vertices = body_model_cpu.forward_kinematics(gt_pose.cpu(), calc_mesh=True)
    J_regressor_batch = J_regressor[None, :].expand(gt_vertices.shape[0], -1, -1)
    gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)[:, :14]
    _, _, vertices = body_model_cpu.forward_kinematics(pose.cpu(), calc_mesh=True)
    keypoints_3d = torch.matmul(J_regressor_batch, vertices)[:, :14]
    pred_pelvis = keypoints_3d[:, [0], :].clone()
    gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
    keypoints_3d = keypoints_3d - pred_pelvis
    gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
    if cal_pampjpe:
        pampjpe = utils.reconstruction_error(keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
        return torch.tensor([(gt_keypoints_3d - keypoints_3d).norm(dim=2).mean(), (gt_vertices - vertices).norm(dim=2).mean(), pampjpe.mean()])
    return torch.tensor([(gt_keypoints_3d - keypoints_3d).norm(dim=2).mean(), (gt_vertices - vertices).norm(dim=2).mean()])

def evaluate_nymeria_metrics(run_smplify=False):
    dataset_path = os.path.join(NYMERIA_DIR, 'test_nymeria.pt')
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(NYMERIA_DIR, 'val_nymeria.pt')
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    print(f"Loading data from {dataset_path}...")
    dataset = torch.load(dataset_path, weights_only=False)

    net = Net().to(device)
    weights_path = os.path.join(paths.weight_dir, Net.name, 'best_weights.pt')
    if os.path.exists(weights_path):
        net.load_state_dict(torch.load(weights_path, weights_only=False))
    net.eval()

    pose_p_list, pose_t_list = [], []
    tran_p_list, tran_t_list = [], []

    print("Running Inference on entire dataset...")
    # Iterate over all sequences
    for seq_idx in tqdm.trange(len(dataset['pose'])):
        # --- Preprocessing (Same as view_nymeria_ours) ---
        accw = dataset['imu_acc'][seq_idx].reshape(-1, 6, 3, 1).to(device)
        oriw = dataset['imu_ori'][seq_idx].reshape(-1, 6, 3, 3).to(device)
        j3dw_mp = dataset['sync_3d_mp'][seq_idx].reshape(-1, 33, 3, 1).to(device)
        j3dw = dataset['joint3d'][seq_idx].reshape(-1, 24, 3, 1).to(device)

       
        root_offset = j3dw[0, 0].clone()
        j3dw = j3dw - root_offset
        j3dw_mp = j3dw_mp - root_offset
        
        # --- Chunked Inference ---
        split_size = 200
        T = len(j3dw)
        
        pose_seq_global = []
        tran_seq_global = []
     
        
        for start in tqdm.tqdm(range(0, T, split_size)):
            end = min(start + split_size, T)
            if end - start < 2: continue # Skip tiny end bits
            
            # 1. Slice Chunk
            accw_chunk = accw[start:end].clone()
            oriw_chunk = oriw[start:end].clone()
            j3dw_mp_chunk = j3dw_mp[start:end].clone()
            j3dw_chunk = j3dw[start:end].clone()
            
            # 2. Re-center relative to Chunk Start
            chunk_root_offset = j3dw_chunk[0, 0].clone()
            j3dw_chunk = j3dw_chunk - chunk_root_offset
            j3dw_mp_chunk = j3dw_mp_chunk - chunk_root_offset
            
            # 3. Virtual Camera Transform (Per Chunk, like training)
            Rcw = get_virtual_camera_transform(fixed_view=True)
            
            # 4. Transform to Camera Frame
            accc_chunk = Rcw.matmul(accw_chunk)
            oric_chunk = Rcw.matmul(oriw_chunk)
            
            j3dw_mp_chunk = j3dw_mp_chunk.reshape(j3dw_mp_chunk.shape[0], 33, 3, 1)
            
            j3dc_chunk = Rcw.matmul(j3dw_chunk).squeeze(-1)
            j3dc_mp_chunk = Rcw.matmul(j3dw_mp_chunk).squeeze(-1)
            
            # 5. Translation & Depth
            random_tranc = get_virtual_translation(j3dc_chunk, fixed_view=True)
            # j3dc_chunk = j3dc_chunk + random_tranc # Not needed for input, only for label/debug
            j3dc_mp_chunk = j3dc_mp_chunk + random_tranc
            
            # 6. Normalize & BBox
            j2dc_chunk = j3dc_mp_chunk / j3dc_mp_chunk[..., -1:]
            j2dc_chunk[..., -1] = 1.0 # Conf
            bbox_scale = get_bbox_scale(j2dc_chunk).view(-1, 1, 1)
            j2dc_chunk[..., :2] = j2dc_chunk[..., :2] / bbox_scale
            j2dc_chunk[:, 24:, :2] = j2dc_chunk[:, 24:, :2] - j2dc_chunk[:, 23:24, :2]
            j2dc_chunk[:, :23, :2] = j2dc_chunk[:, :23, :2] - j2dc_chunk[:, 23:24, :2]
            
            # 7. Network Inputs
            j2dc_input = j2dc_chunk
            accc_input = accc_chunk.squeeze(-1)
            oric_input = oric_chunk
            
            # Update Gravity for this view
            g_world = torch.tensor([0., -1., 0.], device=device).reshape(3, 1)
            g_cam = Rcw.matmul(g_world).view(3)
            Net.gravityc = g_cam
            
            # 8. Run Inference for this Chunk
            net.reset_states() # Must reset because input coordinate system changed!
         
            first_tran_cam = random_tranc 
            
            chunk_pose_preds = []
            chunk_tran_preds = []
            
            for i in range(len(j2dc_input)):
                if i == 0:
                    p, t = net.forward_online(j2dc_input[i], accc_input[i], oric_input[i], first_tran_cam)
                else:
                    p, t = net.forward_online(j2dc_input[i], accc_input[i], oric_input[i])
                chunk_pose_preds.append(p)
                chunk_tran_preds.append(t)
            
            chunk_pose_preds = torch.stack(chunk_pose_preds) # (T_chunk, 24, 3, 3) in Camera Frame
            chunk_tran_preds = torch.stack(chunk_tran_preds) # (T_chunk, 3) in Camera Frame
      
            tran_chunk_world = Rcw.t().matmul((chunk_tran_preds - random_tranc).unsqueeze(-1))
            tran_chunk_global = tran_chunk_world + chunk_root_offset
            tran_chunk_global = tran_chunk_global.squeeze(-1)
            
    
            chunk_pose_global = chunk_pose_preds.clone()
            chunk_pose_global[:, 0] = Rcw.t().matmul(chunk_pose_preds[:, 0])
            
            pose_seq_global.append(chunk_pose_global)
            tran_seq_global.append(tran_chunk_global)
            
        pose_pred = torch.cat(pose_seq_global, dim=0)
        tran_pred = torch.cat(tran_seq_global, dim=0)
        
        # --- End Chunked Inference ---

        # Prepare GT for Evaluation (In World Frame)
        gt_pose_aa = dataset['pose'][seq_idx].to(device)
        gt_pose_rotmat = art.math.axis_angle_to_rotation_matrix(gt_pose_aa).view(-1, 24, 3, 3)

        gt_tran = j3dw[:, 0]

        pose_p_list.append(pose_pred.cpu())
        tran_p_list.append(tran_pred.cpu())
        pose_t_list.append(gt_pose_rotmat.cpu())
        tran_t_list.append(gt_tran.cpu())

    print('Evaluating metrics...')
    errors = torch.stack([cal_mpjpe(pose_p_list[i], pose_t_list[i], cal_pampjpe=True) for i in tqdm.trange(len(pose_p_list))])
    print('mpjpe, pve, pmpjpe:', errors.mean(dim=0))

    eval_fn = art.PositionErrorEvaluator()
    errors_tran = torch.stack([eval_fn(tran_p_list[i], tran_t_list[i]) for i in tqdm.trange(len(tran_p_list))])
    print('absolute root position error:', errors_tran.mean(dim=0))
    
def view_nymeria_unity(seq_idx=0):
    # Minimal version of above loop for Export
    # ... (Loading Logic Same as Above) ...
    dataset_path = os.path.join(NYMERIA_DIR, 'test_nymeria.pt')
    if not os.path.exists(dataset_path): dataset_path = os.path.join(NYMERIA_DIR, 'val_nymeria.pt')
    dataset = torch.load(dataset_path)
    
    accw = dataset['imu_acc'][seq_idx].reshape(-1, 6, 3, 1).to(device)
    oriw = dataset['imu_ori'][seq_idx].reshape(-1, 6, 3, 3).to(device)
    j3dw_mp = dataset['sync_3d_mp'][seq_idx].reshape(-1, 33, 3, 1).to(device)
    j3dw = dataset['joint3d'][seq_idx].reshape(-1, 24, 3, 1).to(device)

    root_offset = j3dw[0, 0].clone()
    j3dw = j3dw - root_offset
    j3dw_mp = j3dw_mp - root_offset
    
    Rcw = get_virtual_camera_transform(fixed_view=True)
    accc = Rcw.matmul(accw) # (T, 6, 3, 1)
    oric = Rcw.matmul(oriw)
    j3dc = Rcw.matmul(j3dw).squeeze(-1)
    j3dc_mp = Rcw.matmul(j3dw_mp).squeeze(-1)
    
    random_tranc = get_virtual_translation(j3dc, fixed_view=True)
    j3dc_mp = j3dc_mp + random_tranc
    j3dc = j3dc + random_tranc
    
    j2dc = j3dc_mp / j3dc_mp[..., -1:]
    j2dc[..., -1] = 1.0 # Conf
    bbox_scale = get_bbox_scale(j2dc).view(-1, 1, 1)
    j2dc[..., :2] = j2dc[..., :2] / bbox_scale
    j2dc[:, 24:, :2] = j2dc[:, 24:, :2] - j2dc[:, 23:24, :2]
    j2dc[:, :23, :2] = j2dc[:, :23, :2] - j2dc[:, 23:24, :2]
    
    j2dc_input = j2dc
    accc_input = accc.squeeze(-1)
    oric_input = oric
    
    g_world = torch.tensor([0., -1., 0.], device=device).reshape(3, 1)
    g_cam = Rcw.matmul(g_world).view(3)
    Net.gravityc = g_cam
    
    net = Net().to(device)
    weights_path = os.path.join(paths.weight_dir, Net.name, 'best_weights.pt')
    if os.path.exists(weights_path):
        net.load_state_dict(torch.load(weights_path))
    net.eval()
    
    pose_pred, tran_pred = [], []
    first_tran_gt = j3dc[0, 0]
    
    for i in tqdm.trange(len(j2dc_input)):
        if i == 0:
            p, t = net.forward_online(j2dc_input[i], accc_input[i], oric_input[i], first_tran_gt)
        else:
            p, t = net.forward_online(j2dc_input[i], accc_input[i], oric_input[i])
        pose_pred.append(p)
        tran_pred.append(t)
        
    pose_pred = torch.stack(pose_pred)
    tran_pred = torch.stack(tran_pred)
    
    tran_world = Rcw.t().matmul((tran_pred - random_tranc).unsqueeze(-1)).squeeze(-1)
    pose_pred[:, 0] = Rcw.t().matmul(pose_pred[:, 0]) # Rotate root orientation back
    
    # Offset to start at 0,0,0
    tran_world = tran_world - tran_world[0]
    
    save_dir = os.path.join(paths.offline_dir, f'nymeria_{seq_idx}_unity')
    os.makedirs(os.path.join(save_dir, '0'), exist_ok=True)
    body_model.save_unity_motion(pose_pred, tran_world, os.path.join(save_dir, '0'))
    print(f"Saved Unity motion to {save_dir}")

if __name__ == '__main__':
    ipdb_safety_net()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('seq_idx', type=int, default=0, nargs='?')
    parser.add_argument('--unity', action='store_true', help='Run unity export')
    parser.add_argument('--eval', action='store_true', help='Run quantitative evaluation')
    args = parser.parse_args()
    
    if args.eval:
        evaluate_nymeria_metrics()
    elif args.unity:
        view_nymeria_unity(args.seq_idx)
    else:
        # Import smplify dynamically
        try:
            from net.smplify.run import smplify_runner
        except ImportError:
            smplify_runner = None
            print("Warning: Could not import smplify_runner. Disabling optimization.")
            
        view_nymeria_ours(args.seq_idx, vis=True, run_smplify=(smplify_runner is not None))
