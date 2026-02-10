
import torch
import cv2
import os
import random
import numpy as np
import tqdm
import articulate as art
import config
from config import *

# Configuration
DATA_PATH = '/home/minghao/src/robotflow/RoHM/third_party/RobustCap/data/dataset_work/TotalCapture/test.pt'
VIDEO_ROOT = '/home/minghao/src/robotflow/RoHM/datasets/TotalCapture/video'
OUTPUT_DIR = '/home/minghao/src/robotflow/RoHM/third_party/RobustCap/verification_output'

# Skeleton definition (HUMBIBody33) from config.py
PARENTS = [
    None,
    0, 0,
    0,
    1, 2,
    3,
    4, 5,
    6,
    7, 8,
    9,
    9, 9,
    12,
    13, 14,
    16, 17,
    18, 19,
    # extended
    15, 15, 15,
    20, 20,
    21, 21,
    7, 7,
    8, 8
]

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
]

def draw_skeleton(img, joints, confidence, threshold=0.1):
    H, W, _ = img.shape
    
    # Draw connections
    for i, p in enumerate(PARENTS):
        if p is not None:
            if confidence[i] > threshold and confidence[p] > threshold:
                pt1 = (int(joints[i, 0]), int(joints[i, 1]))
                pt2 = (int(joints[p, 0]), int(joints[p, 1]))
                cv2.line(img, pt1, pt2, (0, 255, 255), 2)
    
    # Draw joints
    for i in range(len(joints)):
        if confidence[i] > threshold:
            pt = (int(joints[i, 0]), int(joints[i, 1]))
            cv2.circle(img, pt, 4, (0, 0, 255), -1)
            
    return img

def project_3d_to_2d(points_3d, K, T):
    """
    Project 3D points to 2D image plane.
    Args:
        points_3d: (N, 3) 3D points
        K: (3, 3) Intrinsic matrix
        T: (4, 4) Extrinsic matrix (World to Camera)
    Returns:
        points_2d: (N, 2) 2D points
    """
    # Add homogeneous coordinate
    ones = torch.ones(points_3d.shape[0], 1, device=points_3d.device)
    points_3d_h = torch.cat([points_3d, ones], dim=1) # (N, 4)
    
    # Transform to camera coordinate: P_c = T @ P_w
    # T is (4, 4), points_3d_h is (N, 4). We want (T @ P_w^T)^T = P_w @ T^T
    points_cam_h = points_3d_h @ T.t() # (N, 4)
    points_cam = points_cam_h[:, :3] # (N, 3)
    
    # Project to image plane: P_img = K @ P_c
    # K is (3, 3), points_cam is (N, 3). We want (K @ P_c^T)^T = P_c @ K^T
    points_img = points_cam @ K.t() # (N, 3)
    
    # Normalize by Z
    u = points_img[:, 0] / (points_img[:, 2] + 1e-9)
    v = points_img[:, 1] / (points_img[:, 2] + 1e-9)
    
    return torch.stack([u, v], dim=1)

def main():
    print(f"Loading data from {DATA_PATH}...")
    try:
        data = torch.load(DATA_PATH)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    num_samples = len(data['name'])
    print(f"Loaded {num_samples} samples.")

    # Convert tensor lists to lists for easier indexing if needed, but they are lists of tensors usually
    names = data['name']
    joint2d = data['joint2d_minimalbody']
    joint3d = data['joint3d']
    cam_K = data['cam_K']
    cam_T = data['cam_T']
    pose = data['pose']
    tran = data['tran']
    
    # Initialize Body Model
    body_model = art.ParametricModel(paths.smpl_file)

    # Select a random sample
    idx = random.randint(0, num_samples - 1)
    # Or force a specific one if needed for debugging, but random is requested
    # idx = 0 
    
    sample_name = names[idx]
    print(f"Selected sample {idx}: {sample_name}")
    
    # Parse name: TC_S1_acting1 -> Subject: s1, Motion: acting1
    # Note: data['name'] seems to be uppercase S1, but folder is lowercase s1
    parts = sample_name.split('_')
    if len(parts) >= 3:
        subject = parts[1].lower() # S1 -> s1
        motion = parts[2]         # acting1
    else:
        print(f"Error parsing name: {sample_name}")
        return

    print(f"Subject: {subject}, Motion: {motion}")

    # Camera indices (1-8)
    cam_indices = [1, 2, 3, 4, 5, 6, 7, 8]
    selected_cams = random.sample(cam_indices, 6) 
    print(f"Selected cameras: {selected_cams}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for cam_idx in selected_cams:
        # Video path construction
        # Pattern: .../TotalCapture/video/s1/acting1/TC_S1_acting1_cam1.mp4
        # Filename part uses uppercase S1 from sample_name
        video_filename = f"{sample_name}_cam{cam_idx}.mp4"
        video_path = os.path.join(VIDEO_ROOT, subject, motion, video_filename)
        
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            # Try alternative extension if needed? inspection showed .mp4
            continue

        print(f"Processing camera {cam_idx}: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_filename = f"vis_{sample_name}_cam{cam_idx}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        sample_joints = joint2d[idx] 
        sample_joints_3d = joint3d[idx]
        sample_K = cam_K[idx]
        sample_T = cam_T[idx]
        sample_pose = pose[idx]
        sample_tran = tran[idx]

        # camera specific joints: [T, 33, 3]
        cam_joints = sample_joints[cam_idx - 1]
        
        # Camera matrices for this view
        K = sample_K[cam_idx - 1]
        T = sample_T[cam_idx - 1]
        
        num_joint_frames = cam_joints.shape[0]
        
        print(f"Video frames: {total_frames}, Joint frames: {num_joint_frames}")
        
        # Limit frames to verify
        frame_limit = min(300, total_frames, num_joint_frames)
        
        for f in tqdm.tqdm(range(frame_limit)):
            ret, frame = cap.read()
            if not ret:
                break
            
         
            
            # ---------------------------------------------------------
            # Draw 2D Joints (Red) Skeletons (Yellow)
            # ---------------------------------------------------------
            current_joints = cam_joints[f]
            
            for i in range(len(current_joints)):
                x = int(current_joints[i, 0])
                y = int(current_joints[i, 1])
                conf = current_joints[i, 2]
                
                if conf > 0.0: # Threshold
                    # Connections
                    parent = PARENTS[i]
                    if parent is not None:
                        px = int(current_joints[parent, 0])
                        py = int(current_joints[parent, 1])
                        pconf = current_joints[parent, 2]
                        if pconf > 0.0:
                             cv2.line(frame, (x, y), (px, py), (0, 255, 255), 2)
                    
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            # ---------------------------------------------------------
            # NEW: Draw Projected 3D Joints (Blue)
            # ---------------------------------------------------------
            if f < len(sample_joints_3d):
                current_3d = sample_joints_3d[f] # [33, 3]
                
                # Project 3D points to 2D
                current_2d_proj = project_3d_to_2d(current_3d, K, T)
                
                for i in range(len(current_2d_proj)):
                    x_proj = int(current_2d_proj[i, 0].item())
                    y_proj = int(current_2d_proj[i, 1].item())
                    
                    # Check bounds optionally, but cv2 usually handles OOB drawing by clipping
                    # Draw connections
                    parent = PARENTS[i]
                    if parent is not None:
                        px_proj = int(current_2d_proj[parent, 0].item())
                        py_proj = int(current_2d_proj[parent, 1].item())
                        
                        # Draw Line (Blue color: (255, 0, 0) in BGR)
                        cv2.line(frame, (x_proj, y_proj), (px_proj, py_proj), (255, 0, 0), 1)
                    
                    # Draw Joint (Blue)
                    cv2.circle(frame, (x_proj, y_proj), 2, (255, 0, 0), -1)
            # ---------------------------------------------------------

            # ---------------------------------------------------------
            # NEW: Draw FK Joints (Green) and FK Vertices (Cyan)
            # ---------------------------------------------------------
            if f < len(sample_pose):
                # Get current pose and tran
                # pose: [N, 72] or [N, 24, 3] usually axis angle?
                # In preprocess.py: pose = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
                # But data['pose'] might be stored as axis-angle?
                # preprocess.py line 434: newdata['pose'].append(art.math.rotation_matrix_to_axis_angle(pose).view(-1, 24, 3))
                # So it IS axis-angle [N, 24, 3].
                
                curr_pose = sample_pose[f:f+1] # Keep batch dim [1, 24, 3]
                curr_tran = sample_tran[f:f+1] # [1, 3]
                
                # Convert to rotation matrix for body_model
                curr_pose_mat = art.math.axis_angle_to_rotation_matrix(curr_pose).view(-1, 24, 3, 3)
                
                # Forward Kinematics
                # Note: body_model usually requires inputs on same device (cpu here)
                # Ensure float32
                curr_pose_mat = curr_pose_mat.float()
                curr_tran = curr_tran.float()
                
                grot, fk_joint, fk_vert = body_model.forward_kinematics(curr_pose_mat, tran=curr_tran, calc_mesh=True)
                # fk_joint: [1, 24, 3], fk_vert: [1, 6890, 3]
                fk_joint = fk_joint[0]
                fk_vert = fk_vert[0]
                
                # Project FK Joints (Green)
                fk_joint_2d = project_3d_to_2d(fk_joint, K, T)
                for i in range(len(fk_joint_2d)):
                    x_fk = int(fk_joint_2d[i, 0].item())
                    y_fk = int(fk_joint_2d[i, 1].item())
                    cv2.circle(frame, (x_fk, y_fk), 3, (0, 255, 0), -1)
                
                # Project FK Vertices (Cyan) - Downsample for speed/visibility
                # 6890 vertices is a lot, maybe verify TotalCapture uses fewer?
                # No, SMPL usually has 6890.
                # Let's plot every 50th vertex
                fk_vert_sub = fk_vert 
                fk_vert_2d = project_3d_to_2d(fk_vert_sub, K, T)
                
                for i in range(len(fk_vert_2d)):
                    x_v = int(fk_vert_2d[i, 0].item())
                    y_v = int(fk_vert_2d[i, 1].item())
                    cv2.circle(frame, (x_v, y_v), 1, (255, 255, 0), -1)

            # ---------------------------------------------------------

            out.write(frame)
        
        cap.release()
        out.release()
        print(f"Saved visualization to {output_path}")

if __name__ == '__main__':
    main()
