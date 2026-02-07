import articulate as art
import torch
import os
import config
from config import *
import pickle
import tqdm
import json
import glob
import numpy as np

# Placeholder for the root directory of the preprocessed RoHM data
# User should ensure this path is correct or pass it as an argument
ROHM_PREPROCESSED_ROOT = '/home/minghao/src/robotflow/RoHM/third_party/RobustCap/out/AMASS_smplx_preprocessed.robustcap'

# Nymeria splits directory
NYMERIA_SPLITS_DIR = '/home/minghao/src/robotflow/RoHM/datasets/Nymeria_smplx_preprocessed/nymeria_splits'

extkp_mask = torch.tensor(list(HUMBIBody33.extended_keypoints.values()))
body_model = art.ParametricModel(paths.smpl_file)
mp_mask = torch.tensor(config.mp_mask)
vi_mask = torch.tensor(config.vi_mask)
ji_mask = torch.tensor(config.ji_mask)

def _syn_acc(v, smooth_n=2):
    r"""
    Synthesize accelerations from vertex positions.
    """
    mid = smooth_n // 2
    # FPS fixed to 30 as per requirements
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 30 * 30 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 30 * 30 / smooth_n ** 2
             for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc


def process_smplx_sequence(seq_smplx_np, preprocessed_data, body_model, mp_mask, vi_mask, ji_mask, amass_rot, dbg_vis=False):
    """
    Process a single SMPLX sequence and append results to preprocessed_data.
    
    Args:
        seq_smplx_np: numpy array of shape (N, 178) containing SMPLX parameters
        preprocessed_data: dict to append results to
        body_model: SMPL body model for forward kinematics
        mp_mask: MediaPipe vertex mask
        vi_mask: IMU vertex mask
        ji_mask: IMU joint mask
        amass_rot: coordinate alignment rotation matrix
        dbg_vis: bool, whether to visualize the process
    """
    # Ensure float32
    seq_smplx = torch.tensor(seq_smplx_np).float()
    
    N = seq_smplx.shape[0]
    if N < 1:
        return

    # Extract parameters
    global_orient = seq_smplx[:, 0:3]   # (N, 3)
    transl = seq_smplx[:, 3:6]          # (N, 3)
    betas = seq_smplx[:, 6:16]          # (N, 10)
    body_pose_21 = seq_smplx[:, 16:79]  # (N, 63) -> 21 joints
    
    # Construct Pose (N, 24, 3) for SMPL (23 joints + root)
    # RoHM/SMPLX provides 21 body joints. We need to pad 2 wrist joints with identity.
    # Structure: Root (1) + Body (21) + Wrists (2) = 24
    
    pose_final = torch.zeros((N, 24, 3)).float()
    pose_final[:, 0] = global_orient
    pose_final[:, 1:22] = body_pose_21.reshape(N, 21, 3)
    # Indices 22 and 23 remain 0 (Identity in axis-angle)
    
    # Apply coordinate alignment
    # tran alignment
    tran_aligned = amass_rot.matmul(transl.unsqueeze(-1)).view_as(transl)
    
    # pose[:, 0] alignment
    root_rot_mat = art.math.axis_angle_to_rotation_matrix(pose_final[:, 0])  # (N, 3, 3)
    root_rot_aligned = amass_rot.matmul(root_rot_mat)
    pose_final[:, 0] = art.math.rotation_matrix_to_axis_angle(root_rot_aligned)
    
    if N <= 12:
        return

    # Shape is usually consistent per sequence
    shape = betas[0]
    
    p_mat = art.math.axis_angle_to_rotation_matrix(pose_final).view(-1, 24, 3, 3)
    
    # Forward Kinematics
    grot, joint, vert = body_model.forward_kinematics(p_mat, shape=shape, tran=tran_aligned, calc_mesh=True)
    
    if dbg_vis:
        import open3d as o3d
        import time
        print("Debugging Visualization enabled. Visualizing sequence animation...")

        # Prepare geometry data
        verts_np = vert.detach().cpu().numpy()  # (N, V, 3)
        joints_np = joint.detach().cpu().numpy()  # (N, J, 3)
        grots_np = grot.detach().cpu().numpy()  # (N, J, 3, 3)
        faces_np = body_model.face

        # Create Visualizer
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Motion Sequence Visualization", width=1280, height=720)

        # 1. Origin Coordinate Frame
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        vis.add_geometry(origin)

        # 2. Floor Grid (Checkerboard style lines)
        # Create a large grid on XZ plane (y=0)
        def create_floor(size=10, step=1):
            lines = []
            points = []
            # Lines parallel to X-axis
            for z in range(-size, size + 1, step):
                points.append([-size, 0, z])
                points.append([size, 0, z])
                lines.append([len(points) - 2, len(points) - 1])
            # Lines parallel to Z-axis
            for x in range(-size, size + 1, step):
                points.append([x, 0, -size])
                points.append([x, 0, size])
                lines.append([len(points) - 2, len(points) - 1])
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color([0.5, 0.5, 0.5])  # Gray lines
            return line_set

        floor = create_floor()
        vis.add_geometry(floor)

        # 3. Dynamic Elements (Mesh & Joint Frames)
        # Initialize Mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts_np[0])
        mesh.triangles = o3d.utility.Vector3iVector(faces_np)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Gray body
        vis.add_geometry(mesh)

        # Initialize Joint Frames
        joint_frames = []
        num_joints = joints_np.shape[1]
        for j in range(num_joints):
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            frame.rotate(grots_np[0][j], center=(0, 0, 0))
            frame.translate(joints_np[0][j])
            vis.add_geometry(frame)
            joint_frames.append(frame)

        # View Control
        ctr = vis.get_view_control()
        ctr.set_up([0, 1, 0])  # Y-up
        ctr.set_front([0, 0, 1]) # look from +Z
        ctr.set_lookat([0, 0, 0])
        ctr.set_zoom(0.8)

        # Flag to control the animation loop
        keep_running = [True]
        def exit_callback(vis):
            keep_running[0] = False
            return False

        # Register key callbacks for 'Q' and 'q'
        vis.register_key_callback(ord('Q'), exit_callback)
        vis.register_key_callback(ord('q'), exit_callback)

        # Animation Loop
        num_frames = verts_np.shape[0]
        for i in range(num_frames):
            if not keep_running[0]:
                break
            # Update Mesh
            mesh.vertices = o3d.utility.Vector3dVector(verts_np[i])
            mesh.compute_vertex_normals()
            vis.update_geometry(mesh)

            # Update Joint Frames
            # Recreating geometries is slow but easiest for transforms in Open3D python API for non-rigid motions
            # Or we can update vertices of the coordinate frames if we want to be fancy, but simple remove/add works for debug
            # LIMITATION: o3d.visualization.Visualizer.update_geometry for TriangleMesh only updates vertices/normals
            # Coordinate frames are also meshes. But rigid transform is easier via deletion/re-creation or manual transform tracking
            
            # Efficient update: reset transform then apply new one? No, frame doesn't store 'original' state easily.
            # Faster approach: manual vertex transform.
            # Simplest correct approach for debug script: Remove old, Add new.
            
            for j in range(num_joints):
                # Remove old frame
                vis.remove_geometry(joint_frames[j], reset_bounding_box=False)
                
                # Create new frame
                new_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                new_frame.rotate(grots_np[i][j], center=(0, 0, 0))
                new_frame.translate(joints_np[i][j])
                
                vis.add_geometry(new_frame, reset_bounding_box=False)
                joint_frames[j] = new_frame # Update reference

            vis.poll_events()
            vis.update_renderer()
            # time.sleep(0.033) # ~30 FPS

        vis.destroy_window()
    
    # Collect Results
    preprocessed_data['pose'].append(pose_final.clone())       # N, 24, 3
    preprocessed_data['tran'].append(tran_aligned.clone())     # N, 3
    preprocessed_data['shape'].append(shape.clone())           # 10
    preprocessed_data['joint3d'].append(joint)                 # N, J, 3 (from SMPL model)
    preprocessed_data['sync_3d_mp'].append(vert[:, mp_mask])   # N, 33, 3
    
    # IMU synthesis
    preprocessed_data['imu_acc'].append(_syn_acc(vert[:, vi_mask]))
    preprocessed_data['imu_ori'].append(grot[:, ji_mask])


def preprocess_rohm_amass(input_root, out_root, dbg_vis=False):
    print(f'Preprocessing AMASS dataset from RoHM format at {input_root}')
    
    preprocessed_dataset_joints_dir = os.path.join(input_root, 'pose_data_fps_30')
    preprocessed_dataset_smpl_dir = os.path.join(input_root, 'smpl_data_fps_30')

    # AMASS global alignment rotation (from original preprocess.py)
    # Align AMASS global fame with AIST
    amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]]).float()

    # Iterate over splits defined in config.amass_data
    # Note: config.amass_data usually has 'val' and 'train'. 'test' might be empty but we check anyway.
    for kind in ['val', 'train', 'test']:
        dataset_names = getattr(amass_data, kind)
        if not dataset_names:
            continue
            
        print(f'Processing {kind} split...')
        
        preprocessed_data = {'pose': [], 'shape': [], 'tran': [], 'joint3d': [], 'imu_ori': [], 'imu_acc': [], 'sync_3d_mp': []}
        
        for ds_name in dataset_names:
            print(f'\rReading {ds_name}', end='')
            
            # Match files similar to dataset_amass_like.py
            # Using recursive glob pattern from dataset_amass_like.py: os.path.join(dir, dataset_name, '**/*.npy')
            seq_joints_paths = sorted(glob.glob(os.path.join(preprocessed_dataset_joints_dir, ds_name, '**/*.npy'), recursive=True))
            seq_smpl_paths = sorted(glob.glob(os.path.join(preprocessed_dataset_smpl_dir, ds_name, '**/*.npy'), recursive=True))
            
            if len(seq_joints_paths) != len(seq_smpl_paths):
                print(f'\nWarning: Mismatch in file counts for {ds_name} ({len(seq_joints_paths)} joints vs {len(seq_smpl_paths)} smpl). Skipping mismatched tail.')
            
            # Assuming strictly sorted 1-to-1 mapping
            for seq_joint_path, seq_smpl_path in zip(seq_joints_paths, seq_smpl_paths):
                try:
                    # Load SMPLX data: [seq_len, 178]
                    seq_smplx = np.load(seq_smpl_path)
                    process_smplx_sequence(seq_smplx, preprocessed_data, body_model, mp_mask, vi_mask, ji_mask, amass_rot, dbg_vis=dbg_vis)
                    
                except Exception as e:
                    print(f'\nError processing {seq_smpl_path}: {e}')
                    continue

        print(f'\nSaving {kind}_rohm.pt...')
        os.makedirs(out_root, exist_ok=True)
        torch.save(preprocessed_data, os.path.join(out_root, f'{kind}_rohm.pt'))
        print('Done.')


def preprocess_rohm_nymeria(input_root, out_root, dbg_vis=False):
    print(f'Preprocessing Nymeria dataset from RoHM format at {input_root}')

    splits_dir = os.path.join(input_root, "nymeria_splits")
    preprocessed_dataset_joints_dir = os.path.join(input_root, 'pose_data_fps_30', 'Nymeria')
    preprocessed_dataset_smpl_dir = os.path.join(input_root, 'smpl_data_fps_30', 'Nymeria')
    
    # AMASS global alignment rotation
    amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]]).float()

    for kind in ['train', 'val', 'test']:
        split_file = os.path.join(splits_dir, f'{kind}.txt')
        if not os.path.exists(split_file):
            print(f'Split file {split_file} not found, skipping {kind}.')
            continue
            
        print(f'Processing {kind} split...')
        
        with open(split_file, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
            
        preprocessed_data = {'pose': [], 'shape': [], 'tran': [], 'joint3d': [], 'imu_ori': [], 'imu_acc': [], 'sync_3d_mp': []}
        
        for seq_name in tqdm.tqdm(sequences):
            # Locate files for this sequence
            # Files are in input_root/smpl_data_fps_30/Nymeria/{seq_name}/*.npy
            
            seq_dir_smpl = os.path.join(preprocessed_dataset_smpl_dir, seq_name)
            seq_dir_joints = os.path.join(preprocessed_dataset_joints_dir, seq_name)
            
            if not os.path.exists(seq_dir_smpl):
                print(f'\nWarning: Sequence dir {seq_dir_smpl} not found.')
                continue
                
            seq_smpl_paths = sorted(glob.glob(os.path.join(seq_dir_smpl, '*.npy')))
            
            if len(seq_smpl_paths) == 0:
                print(f'\nWarning: No npy files found in {seq_dir_smpl}')
                continue
                
            # Loop through npy files in sequence folder (usually chunked or single file)
            for seq_smpl_path in seq_smpl_paths:
                try:
                    seq_smplx = np.load(seq_smpl_path)
                    process_smplx_sequence(seq_smplx, preprocessed_data, body_model, mp_mask, vi_mask, ji_mask, amass_rot, dbg_vis=dbg_vis)
                except Exception as e:
                    print(f'\nError processing {seq_smpl_path}: {e}')
                    continue

        print(f'\nSaving {kind}_nymeria.pt...')
        os.makedirs(out_root, exist_ok=True)
        torch.save(preprocessed_data, os.path.join(out_root, f'{kind}_nymeria.pt'))
        print('Done.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='all', choices=['amass', 'nymeria', 'all'])
    parser.add_argument('--dbg_vis', action='store_true', help='Enable visualization for debugging')
    args = parser.parse_args()
    
    if args.dataset in ['amass', 'all']:
        preprocess_rohm_amass(args.input_root, args.output_root, dbg_vis=args.dbg_vis)
    if args.dataset in ['nymeria', 'all']:
        preprocess_rohm_nymeria(args.input_root, args.output_root, dbg_vis=args.dbg_vis)
