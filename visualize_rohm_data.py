import articulate as art
import torch
import os
import config
from config import *
import numpy as np
import open3d as o3d
import argparse
import time

def visualize_sequence(verts_np, joints_np, grots_np, faces_np):
    """
    Visualizes a single sequence of motion.
    Args:
        verts_np: (N, V, 3)
        joints_np: (N, J, 3)
        grots_np: (N, J, 3, 3)
        faces_np: (F, 3)
    """
    print("Visualizing sequence animation...")

    # Create Visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Motion Sequence Visualization", width=1280, height=720)

    # 1. Origin Coordinate Frame
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(origin)

    # 2. Floor Grid (Checkerboard style lines)
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
    exit_script = [False]
    
    def next_seq_callback(vis):
        keep_running[0] = False
        return False
        
    def exit_callback(vis):
        keep_running[0] = False
        exit_script[0] = True
        return False

    # Register key callbacks
    vis.register_key_callback(ord('Q'), next_seq_callback)
    vis.register_key_callback(ord('q'), next_seq_callback)
    vis.register_key_callback(27, exit_callback) # ESC to exit completely

    # Animation Loop
    num_frames = verts_np.shape[0]
    for i in range(num_frames):
        if not keep_running[0]:
            break
        # Update Mesh
        mesh.vertices = o3d.utility.Vector3dVector(verts_np[i])
        mesh.compute_vertex_normals()
        vis.update_geometry(mesh)

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
        time.sleep(0.02) # ~50 FPS cap

    vis.destroy_window()
    return exit_script[0]

def main():
    parser = argparse.ArgumentParser(description="Visualize RoHM processed data (.pt files)")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the .pt file")
    parser.add_argument('--start_idx', type=int, default=0, help="Start index of sequence to visualize")
    parser.add_argument('--count', type=int, default=None, help="Number of sequences to visualize")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File {args.file_path} not found.")
        return

    print(f"Loading data from {args.file_path}...")
    data = torch.load(args.file_path)
    
    # RoHM data format is usually a dict of lists
    # Keys might include 'pose', 'tran', 'shape', 'joint3d', 'imu_ori', 'imu_acc', 'sync_3d_mp'
    
    if 'pose' not in data or 'tran' not in data or 'shape' not in data:
        print("Error: Missing required keys (pose, tran, shape) in data.")
        print(f"Available keys: {data.keys()}")
        return

    num_sequences = len(data['pose'])
    print(f"Found {num_sequences} sequences.")

    # Initialize body model
    body_model = art.ParametricModel(paths.smpl_file)

    start_idx = args.start_idx
    end_idx = num_sequences
    if args.count is not None:
        end_idx = min(start_idx + args.count, num_sequences)

    for i in range(start_idx, end_idx):
        print(f"Processing sequence {i}/{num_sequences}...")
        
        pose = data['pose'][i] # (N, 24, 3)
        tran = data['tran'][i] # (N, 3)
        shape = data['shape'][i] # (10,)
        
        # Ensure correct shapes and types
        if len(pose.shape) == 2: # Check if flattened
             # Assuming standard SMPL 24 joints * 3 = 72
             if pose.shape[1] == 72:
                 pose = pose.view(-1, 24, 3)
        
        N = pose.shape[0]
        if N == 0:
            print(f"Sequence {i} is empty. Skipping.")
            continue
            
        # Forward Kinematics
        print("Running Forward Kinematics to get mesh...")
        # Prepare inputs for body_model
        # body_model.forward_kinematics expects rotation matrices usually if not specified otherwise in art?
        # Checking preprocess script: 
        # p_mat = art.math.axis_angle_to_rotation_matrix(pose_final).view(-1, 24, 3, 3)
        # grot, joint, vert = body_model.forward_kinematics(p_mat, shape=shape, tran=tran_aligned, calc_mesh=True)
        
        p_mat = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
        
        # Move to same device as model (CPU usually for vis)
        p_mat = p_mat.to(torch.float32)
        shape = shape.to(torch.float32)
        tran = tran.to(torch.float32)
        
        with torch.no_grad():
            grot, joint, vert = body_model.forward_kinematics(p_mat, shape=shape, tran=tran, calc_mesh=True)
        
        verts_np = vert.detach().cpu().numpy()
        joints_np = joint.detach().cpu().numpy()
        grots_np = grot.detach().cpu().numpy()
        faces_np = body_model.face
        
        should_exit = visualize_sequence(verts_np, joints_np, grots_np, faces_np)
        
        if should_exit:
            print("Exiting visualization.")
            break

if __name__ == "__main__":
    main()
