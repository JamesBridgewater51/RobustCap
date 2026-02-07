#!/usr/bin/env python3
"""
Dataset Audit Script for Nymeria vs AMASS Comparison

This script compares the statistical properties of Nymeria and AMASS datasets
to identify root causes for training loss discrepancies.

Usage:
    python tools/audit_dataset.py --nymeria-path /path/to/nymeria --amass-path /path/to/amass

Author: Dataset Audit Tool
"""

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class IMUStats:
    """Statistics for IMU sensor data."""
    acc_mean: np.ndarray  # (6, 3) mean acceleration per sensor
    acc_std: np.ndarray   # (6, 3) std acceleration per sensor
    acc_min: np.ndarray   # (6, 3) min values
    acc_max: np.ndarray   # (6, 3) max values
    acc_magnitude_mean: float
    acc_magnitude_std: float
    
    ori_gravity_alignment: float  # alignment with gravity vector


@dataclass
class JointStats:
    """Statistics for 3D joint data."""
    bone_lengths_mean: np.ndarray   # (23,) mean bone lengths
    bone_lengths_std: np.ndarray    # (23,) std (should be ~0 for same body)
    joint_velocity_mean: float
    joint_velocity_std: float
    joint_velocity_percentiles: np.ndarray  # [25, 50, 75, 95, 99]
    
    root_height_mean: float
    root_height_std: float
    floor_penetration_rate: float   # % of frames with joints below z=0
    
    pose_range: np.ndarray  # range of pose values per joint

    # New fields for distribution analysis
    coord_mean: np.ndarray  # (3,) Mean X, Y, Z coordinates
    coord_std: np.ndarray   # (3,) Std dev of X, Y, Z coordinates
    coord_min: np.ndarray   # (3,) Min X, Y, Z coordinates
    coord_max: np.ndarray   # (3,) Max X, Y, Z coordinates
    abs_mean: float         # Mean absolute value of all coordinates
    global_extent: float    # Max distance between any two points in the dataset (approx scale)
    
    # Internal usage for visualization
    _sample_joints: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


@dataclass
class DataScaleStats:
    """Statistics for dataset scale and composition."""
    total_sequences: int
    total_frames: int
    avg_sequence_length: float
    min_sequence_length: int
    max_sequence_length: int
    
    has_2d_keypoints: bool
    has_real_camera: bool
    num_camera_views: int


@dataclass
class DiversityStats:
    """Statistics for motion diversity."""
    pose_pca_explained_variance: np.ndarray  # first 10 components
    velocity_entropy: float
    action_coverage: float  # estimated from pose space


@dataclass
class DatasetAuditResult:
    """Complete audit result for a dataset."""
    name: str
    imu: IMUStats
    joints: JointStats
    scale: DataScaleStats
    diversity: DiversityStats


def load_amass_dataset(data_dir: str, kind: str = 'train') -> Dict:
    """Load AMASS preprocessed dataset."""
    path = os.path.join(data_dir, f'{kind}.pt')
    print(f'Loading AMASS {kind} from {path}')
    return torch.load(path)


def load_nymeria_dataset(data_dir: str, kind: str = 'train') -> Dict:
    """Load Nymeria preprocessed dataset."""
    path = os.path.join(data_dir, f'{kind}_nymeria.pt')
    print(f'Loading Nymeria {kind} from {path}')
    return torch.load(path)


def compute_imu_stats(dataset: Dict) -> IMUStats:
    """Compute IMU-related statistics."""
    # Collect all accelerations
    all_acc = []
    all_ori = []
    
    key_acc = 'imu_acc' if 'imu_acc' in dataset else None
    key_ori = 'imu_ori' if 'imu_ori' in dataset else None
    
    if key_acc is None:
        print("Warning: No IMU acceleration data found")
        return IMUStats(
            acc_mean=np.zeros((6, 3)),
            acc_std=np.zeros((6, 3)),
            acc_min=np.zeros((6, 3)),
            acc_max=np.zeros((6, 3)),
            acc_magnitude_mean=0.0,
            acc_magnitude_std=0.0,
            ori_gravity_alignment=0.0
        )
    
    for i in tqdm(range(len(dataset[key_acc])), desc="Computing IMU stats"):
        acc = dataset[key_acc][i]
        if isinstance(acc, torch.Tensor):
            acc = acc.numpy()
        all_acc.append(acc)
        
        if key_ori is not None:
            ori = dataset[key_ori][i]
            if isinstance(ori, torch.Tensor):
                ori = ori.numpy()
            all_ori.append(ori)
    
    # Stack all data
    if len(all_acc) > 0:
        all_acc = np.concatenate(all_acc, axis=0)  # (N, 6, 3)
        
        # Compute per-sensor statistics
        acc_mean = all_acc.mean(axis=0)  # (6, 3)
        acc_std = all_acc.std(axis=0)
        acc_min = all_acc.min(axis=0)
        acc_max = all_acc.max(axis=0)
        
        # Magnitude statistics
        acc_magnitude = np.linalg.norm(all_acc, axis=-1)  # (N, 6)
        acc_magnitude_mean = acc_magnitude.mean()
        acc_magnitude_std = acc_magnitude.std()
    else:
        acc_mean = np.zeros((6, 3))
        acc_std = np.zeros((6, 3))
        acc_min = np.zeros((6, 3))
        acc_max = np.zeros((6, 3))
        acc_magnitude_mean = 0.0
        acc_magnitude_std = 0.0

    # Gravity alignment (check if root IMU aligns with expected gravity)
    # Assuming Z-up or Y-up, gravity should be ~9.8 in one axis
    gravity_alignment = 0.0
    
    if len(all_ori) > 0:
        all_ori = np.concatenate(all_ori, axis=0)  # (N, 6, 3, 3)
        # Check orientation consistency
        if all_ori.shape[-1] == 3 and all_ori.shape[-2] == 3:
             gravity_alignment = np.abs(all_ori[:, -1, :, 2].mean(axis=0)).max()
    
    return IMUStats(
        acc_mean=acc_mean,
        acc_std=acc_std,
        acc_min=acc_min,
        acc_max=acc_max,
        acc_magnitude_mean=acc_magnitude_mean,
        acc_magnitude_std=acc_magnitude_std,
        ori_gravity_alignment=gravity_alignment
    )


def compute_joint_stats(dataset: Dict) -> JointStats:
    """Compute 3D joint-related statistics."""
    key_joint = 'joint3d' if 'joint3d' in dataset else None
    key_pose = 'pose' if 'pose' in dataset else None
    
    if key_joint is None:
        print("Warning: No joint3d data found")
        return JointStats(
            bone_lengths_mean=np.zeros(23),
            bone_lengths_std=np.zeros(23),
            joint_velocity_mean=0.0,
            joint_velocity_std=0.0,
            joint_velocity_percentiles=np.zeros(5),
            root_height_mean=0.0,
            root_height_std=0.0,
            floor_penetration_rate=0.0,
            pose_range=np.zeros((24, 3)),
            coord_mean=np.zeros(3),
            coord_std=np.zeros(3),
            coord_min=np.zeros(3),
            coord_max=np.zeros(3),
            abs_mean=0.0,
            global_extent=0.0
        )
    
    # SMPL skeleton parent indices
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    
    all_bone_lengths = []
    all_velocities = []
    all_root_heights = []
    floor_penetration_count = 0
    total_frames = 0
    all_poses = []
    
    # Distribution stats aggregators
    sample_accumulated_joints = [] 
    
    running_sum = np.zeros(3)
    running_sq_sum = np.zeros(3)
    running_abs_sum = 0.0
    running_min = np.full(3, np.inf)
    running_max = np.full(3, -np.inf)
    total_joint_points = 0
    
    SUBSAMPLE_RATE = 10 
    
    for i in tqdm(range(len(dataset[key_joint])), desc="Computing joint stats"):
        joints = dataset[key_joint][i]
        if isinstance(joints, torch.Tensor):
            joints = joints.numpy()
        
        T, J, C = joints.shape
        # Typically J=24, C=3
        
        # Basic stats
        bone_lengths = []
        for j in range(1, min(24, joints.shape[1])):
            if j < len(parents) and parents[j] >= 0:
                bone_len = np.linalg.norm(
                    joints[:, j] - joints[:, parents[j]], axis=-1
                )
                bone_lengths.append(bone_len.mean())
        all_bone_lengths.append(bone_lengths)
        
        if joints.shape[0] > 1:
            velocity = np.linalg.norm(joints[1:] - joints[:-1], axis=-1)
            all_velocities.append(velocity.flatten())
        
        root_heights = joints[:, 0, 2]  # Z coordinate of root
        all_root_heights.append(root_heights)
        
        min_z = joints[:, :, 2].min(axis=1)
        floor_penetration_count += (min_z < 0).sum()
        total_frames += joints.shape[0]
        
        if key_pose is not None:
            pose = dataset[key_pose][i]
            if isinstance(pose, torch.Tensor):
                pose = pose.numpy()
            all_poses.append(pose)
            
        # Distribution stats
        flattened_joints = joints.reshape(-1, 3) # (T*J, 3)
        
        total_joint_points += flattened_joints.shape[0]
        running_sum += flattened_joints.sum(axis=0)
        running_sq_sum += (flattened_joints ** 2).sum(axis=0)
        running_abs_sum += np.abs(flattened_joints).sum()
        
        batch_min = flattened_joints.min(axis=0)
        batch_max = flattened_joints.max(axis=0)
        running_min = np.minimum(running_min, batch_min)
        running_max = np.maximum(running_max, batch_max)

        if i % SUBSAMPLE_RATE == 0:
            # Subsample frames within sequence too
            sample_accumulated_joints.append(flattened_joints[::20]) 
    
    # Aggregate stats
    all_bone_lengths = np.array(all_bone_lengths)
    if len(all_bone_lengths) > 0:
        bone_lengths_mean = all_bone_lengths.mean(axis=0)
        bone_lengths_std = all_bone_lengths.std(axis=0)
    else:
        bone_lengths_mean = np.zeros(23)
        bone_lengths_std = np.zeros(23)
        
    if len(all_velocities) > 0:
        all_velocities = np.concatenate(all_velocities)
        velocity_mean = all_velocities.mean()
        velocity_std = all_velocities.std()
        velocity_percentiles = np.percentile(all_velocities, [25, 50, 75, 95, 99])
    else:
        velocity_mean = 0.0
        velocity_std = 0.0
        velocity_percentiles = np.zeros(5)
    
    if len(all_root_heights) > 0:
        all_root_heights = np.concatenate(all_root_heights)
        root_height_mean = all_root_heights.mean()
        root_height_std = all_root_heights.std()
    else:
        root_height_mean = 0.0
        root_height_std = 0.0
        
    pose_range = np.zeros((24, 3))
    if len(all_poses) > 0:
        all_poses = np.concatenate(all_poses, axis=0)
        if len(all_poses.shape) == 2:
             if all_poses.shape[1] == 72:
                 all_poses = all_poses.reshape(-1, 24, 3)
             elif all_poses.shape[1] % 3 == 0:
                 all_poses = all_poses.reshape(-1, all_poses.shape[1]//3, 3)
        if len(all_poses.shape) == 3 and all_poses.shape[1] == 24:
            pose_range = all_poses.max(axis=0) - all_poses.min(axis=0)

    # Coords final calc
    if total_joint_points > 0:
        coord_mean = running_sum / total_joint_points
        mean_sq = running_sq_sum / total_joint_points
        # Avoid negative due to float precision
        coord_std = np.sqrt(np.maximum(mean_sq - coord_mean**2, 0)) 
        abs_mean = running_abs_sum / total_joint_points
        
        sample_joints = np.concatenate(sample_accumulated_joints, axis=0) if sample_accumulated_joints else np.zeros((0,3))
    else:
        coord_mean = np.zeros(3)
        coord_std = np.zeros(3)
        running_min = np.zeros(3)
        running_max = np.zeros(3)
        abs_mean = 0.0
        sample_joints = np.zeros((0, 3))

    global_extent = np.linalg.norm(running_max - running_min)
    
    # Store samples in hidden field for visualizer
    # Using field(default=...) in dataclass is static, but we return object instance
    stats = JointStats(
        bone_lengths_mean=bone_lengths_mean,
        bone_lengths_std=bone_lengths_std,
        joint_velocity_mean=velocity_mean,
        joint_velocity_std=velocity_std,
        joint_velocity_percentiles=velocity_percentiles,
        root_height_mean=root_height_mean,
        root_height_std=root_height_std,
        floor_penetration_rate=floor_penetration_count / max(total_frames, 1),
        pose_range=pose_range,
        coord_mean=coord_mean,
        coord_std=coord_std,
        coord_min=running_min,
        coord_max=running_max,
        abs_mean=abs_mean,
        global_extent=global_extent,
        _sample_joints=sample_joints
    )
    return stats


def compute_scale_stats(dataset: Dict) -> DataScaleStats:
    """Compute dataset scale statistics."""
    # Find a key that represents sequences
    seq_key = None
    for key in ['imu_acc', 'pose', 'joint3d', 'tran']:
        if key in dataset:
            seq_key = key
            break
    
    if seq_key is None:
        return DataScaleStats(
            total_sequences=0,
            total_frames=0,
            avg_sequence_length=0,
            min_sequence_length=0,
            max_sequence_length=0,
            has_2d_keypoints=False,
            has_real_camera=False,
            num_camera_views=0
        )
    
    total_sequences = len(dataset[seq_key])
    
    lengths = []
    for i in range(total_sequences):
        data = dataset[seq_key][i]
        if isinstance(data, torch.Tensor):
            lengths.append(data.shape[0])
        else:
            lengths.append(len(data))
    
    lengths = np.array(lengths)
    total_frames = lengths.sum()
    
    has_2d = 'joint2d_mp' in dataset or 'joint2d' in dataset
    has_cam = 'cam_K' in dataset or 'cam_T' in dataset
    
    num_views = 0
    if 'joint2d_mp' in dataset and len(dataset['joint2d_mp']) > 0:
        sample = dataset['joint2d_mp'][0]
        if isinstance(sample, list):
            num_views = len([x for x in sample if x is not None])
    
    return DataScaleStats(
        total_sequences=total_sequences,
        total_frames=int(total_frames),
        avg_sequence_length=lengths.mean(),
        min_sequence_length=int(lengths.min()),
        max_sequence_length=int(lengths.max()),
        has_2d_keypoints=has_2d,
        has_real_camera=has_cam,
        num_camera_views=num_views
    )


def compute_diversity_stats(dataset: Dict) -> DiversityStats:
    """Compute motion diversity statistics."""
    from sklearn.decomposition import PCA
    
    key_pose = 'pose' if 'pose' in dataset else None
    
    if key_pose is None:
        return DiversityStats(
            pose_pca_explained_variance=np.zeros(10),
            velocity_entropy=0.0,
            action_coverage=0.0
        )
    
    all_poses = []
    for i in range(len(dataset[key_pose])):
        pose = dataset[key_pose][i]
        if isinstance(pose, torch.Tensor):
            pose = pose.numpy()
        if len(pose.shape) == 3:
            pose = pose.reshape(pose.shape[0], -1)
        all_poses.append(pose)
    
    all_poses = np.concatenate(all_poses, axis=0)
    
    if len(all_poses) > 100000:
        indices = np.random.choice(len(all_poses), 100000, replace=False)
        all_poses = all_poses[indices]
    
    n_components = min(10, all_poses.shape[1], len(all_poses))
    pca = PCA(n_components=n_components)
    pca.fit(all_poses)
    
    velocities = np.linalg.norm(all_poses[1:] - all_poses[:-1], axis=-1)
    velocities = velocities[velocities > 1e-5] 
    
    if len(velocities) > 0:
        hist, _ = np.histogram(velocities, bins=50, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist))
    else:
        entropy = 0.0
    
    return DiversityStats(
        pose_pca_explained_variance=pca.explained_variance_ratio_,
        velocity_entropy=entropy,
        action_coverage=pca.explained_variance_ratio_[:3].sum()
    )


def audit_dataset(data_dir: str, name: str, kind: str = 'train', 
                  is_nymeria: bool = False) -> DatasetAuditResult:
    """Run complete audit on a dataset."""
    print(f"\n{'='*60}")
    print(f"Auditing {name} ({kind})")
    print(f"{'='*60}")
    
    # Load dataset
    if is_nymeria:
        dataset = load_nymeria_dataset(data_dir, kind)
    else:
        dataset = load_amass_dataset(data_dir, kind)
    
    print(f"Dataset keys: {list(dataset.keys())}")
    
    imu_stats = compute_imu_stats(dataset)
    joint_stats = compute_joint_stats(dataset)
    scale_stats = compute_scale_stats(dataset)
    diversity_stats = compute_diversity_stats(dataset)
    
    return DatasetAuditResult(
        name=name,
        imu=imu_stats,
        joints=joint_stats,
        scale=scale_stats,
        diversity=diversity_stats
    )


def compare_datasets(result_a: DatasetAuditResult, 
                     result_b: DatasetAuditResult) -> str:
    """Generate comparison report between two datasets."""
    lines = []
    lines.append(f"\n# Dataset Comparison: {result_a.name} vs {result_b.name}")
    lines.append("=" * 60)
    
    # Scale comparison
    lines.append("\n## 📊 Scale Comparison")
    lines.append(f"| Metric | {result_a.name} | {result_b.name} | Ratio |")
    lines.append("|--------|--------|--------|-------|")
    
    ratio_seq = result_b.scale.total_sequences / max(result_a.scale.total_sequences, 1)
    ratio_frames = result_b.scale.total_frames / max(result_a.scale.total_frames, 1)
    
    lines.append(f"| Sequences | {result_a.scale.total_sequences:,} | {result_b.scale.total_sequences:,} | {ratio_seq:.2f}x |")
    lines.append(f"| Frames | {result_a.scale.total_frames:,} | {result_b.scale.total_frames:,} | {ratio_frames:.2f}x |")
    lines.append(f"| Avg Length | {result_a.scale.avg_sequence_length:.1f} | {result_b.scale.avg_sequence_length:.1f} | - |")
    
    # IMU comparison
    lines.append("\n## 🎯 IMU Statistics")
    
    acc_mean_diff = np.abs(result_a.imu.acc_mean - result_b.imu.acc_mean).max()
    
    lines.append(f"| Metric | {result_a.name} | {result_b.name} |")
    lines.append("|--------|--------|--------|")
    lines.append(f"| Acc Magnitude Mean | {result_a.imu.acc_magnitude_mean:.2f} | {result_b.imu.acc_magnitude_mean:.2f} |")
    lines.append(f"| Acc Magnitude Std | {result_a.imu.acc_magnitude_std:.2f} | {result_b.imu.acc_magnitude_std:.2f} |")
    lines.append(f"| Max Mean Difference | - | {acc_mean_diff:.4f} |")
    
    if acc_mean_diff > 100:
        lines.append("\n> ⚠️ **WARNING**: Large IMU acceleration difference detected!")
    
    # Joint comparison
    lines.append("\n## 🦴 Joint Statistics (Scale & Distribution)")
    
    bone_diff = np.abs(result_a.joints.bone_lengths_mean - result_b.joints.bone_lengths_mean).mean()
    abs_mean_ratio = result_b.joints.abs_mean / max(result_a.joints.abs_mean, 1e-6)
    extent_ratio = result_b.joints.global_extent / max(result_a.joints.global_extent, 1e-6)
    
    lines.append(f"| Metric | {result_a.name} | {result_b.name} | Ratio |")
    lines.append("|--------|--------|--------|-------|")
    lines.append(f"| Global Coord Mean (val) | {result_a.joints.abs_mean:.4f} | {result_b.joints.abs_mean:.4f} | {abs_mean_ratio:.2f}x |")
    lines.append(f"| Global Extent (max-min) | {result_a.joints.global_extent:.4f} | {result_b.joints.global_extent:.4f} | {extent_ratio:.2f}x |")
    lines.append(f"| Mean Bone Diff | - | {bone_diff:.4f} | - |")
    lines.append(f"| Velocity Mean | {result_a.joints.joint_velocity_mean:.4f} | {result_b.joints.joint_velocity_mean:.4f} | - |")
    lines.append(f"| Floor Penetration | {result_a.joints.floor_penetration_rate:.2%} | {result_b.joints.floor_penetration_rate:.2%} | - |")
    lines.append(f"| Root Height Mean | {result_a.joints.root_height_mean:.3f} | {result_b.joints.root_height_mean:.3f} | - |")
    
    if bone_diff > 0.1:
        lines.append("\n> ⚠️ **WARNING**: Significant bone length difference!")
    if abs_mean_ratio > 50 or abs_mean_ratio < 0.02:
         lines.append(f"\n> ⚠️ **CRITICAL WARNING**: Huge Joint Coordinate Scale Difference ({abs_mean_ratio:.2f}x)! Check units (mm vs m)!")

    # Diversity comparison
    lines.append("\n## 🎨 Diversity Statistics")
    lines.append(f"| Metric | {result_a.name} | {result_b.name} |")
    lines.append("|--------|--------|--------|")
    lines.append(f"| PCA Top-3 Coverage | {result_a.diversity.action_coverage:.2%} | {result_b.diversity.action_coverage:.2%} |")
    lines.append(f"| Velocity Entropy | {result_a.diversity.velocity_entropy:.4f} | {result_b.diversity.velocity_entropy:.4f} |")
    
    # Conclusions
    lines.append("\n## 💡 Key Findings")
    findings = []
    
    if acc_mean_diff > 100:
        findings.append("- **IMU Scale Mismatch**: Acceleration values differ significantly")
    
    if bone_diff > 0.05:
        findings.append("- **Skeleton Mismatch**: Bone lengths differ between datasets")
    
    if result_b.joints.floor_penetration_rate > 0.01:
        findings.append("- **Floor Issues**: Significant floor penetration detected")
    
    if abs(result_a.joints.root_height_mean - result_b.joints.root_height_mean) > 0.5:
        findings.append("- **Coordinate System**: Root heights differ (possible Y-up vs Z-up)")
    
    if abs_mean_ratio > 10.0:
        findings.append(f"- **SCALE MISMATCH**: Data B is {abs_mean_ratio:.1f}x larger than Data A in joint coordinates!")
        
    if not findings:
        findings.append("- No critical issues detected based on statistics")
    
    lines.extend(findings)
    
    return "\n".join(lines)


def visualize_comparison(result_a: DatasetAuditResult, 
                         result_b: DatasetAuditResult,
                         output_dir: str):
    """Generate visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3)
    
    # 1. IMU acceleration magnitude distribution
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(6)
    width = 0.35
    # Plotting Norm of Mean Acceleration per sensor
    ax1.bar(x - width/2, np.linalg.norm(result_a.imu.acc_mean, axis=1), width, label=result_a.name, alpha=0.8)
    ax1.bar(x + width/2, np.linalg.norm(result_b.imu.acc_mean, axis=1), width, label=result_b.name, alpha=0.8)
    ax1.set_xlabel('IMU Sensor Index')
    ax1.set_ylabel('Norm of Mean Acc')
    ax1.set_title('IMU Mean Acceleration Magnitude')
    ax1.legend()
    
    # 2. Bone length comparison
    ax2 = fig.add_subplot(gs[0, 1])
    n_bones = min(len(result_a.joints.bone_lengths_mean), len(result_b.joints.bone_lengths_mean))
    x = np.arange(n_bones)
    ax2.bar(x - width/2, result_a.joints.bone_lengths_mean[:n_bones], width, label=result_a.name, alpha=0.8)
    ax2.bar(x + width/2, result_b.joints.bone_lengths_mean[:n_bones], width, label=result_b.name, alpha=0.8)
    ax2.set_xlabel('Bone Index')
    ax2.set_ylabel('Length (m)')
    ax2.set_title('Bone Lengths')
    ax2.legend()
    
    # 3. Velocity percentiles
    ax3 = fig.add_subplot(gs[0, 2])
    percentiles = [25, 50, 75, 95, 99]
    x = np.arange(len(percentiles))
    ax3.bar(x - width/2, result_a.joints.joint_velocity_percentiles, width, label=result_a.name, alpha=0.8)
    ax3.bar(x + width/2, result_b.joints.joint_velocity_percentiles, width, label=result_b.name, alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{p}%' for p in percentiles])
    ax3.set_title('Joint Velocity Distribution')
    ax3.legend()
    
    # 4. Global Joint Coordinate Density
    ax4 = fig.add_subplot(gs[1, :])
    samples_a = result_a.joints._sample_joints.flatten()
    samples_b = result_b.joints._sample_joints.flatten()
    
    # Dynamic range for hist based on data
    all_samples = np.concatenate([samples_a, samples_b])
    if len(all_samples) > 0:
        lim = np.percentile(np.abs(all_samples), 99)
        hist_range = (-lim, lim)
    else:
        hist_range = (-2, 2)
        
    ax4.hist(samples_a, bins=100, density=True, alpha=0.5, label=result_a.name, color='blue', range=hist_range)
    ax4.hist(samples_b, bins=100, density=True, alpha=0.5, label=result_b.name, color='orange', range=hist_range)
    ax4.set_title('Global Joint Coordinate Distribution')
    ax4.set_xlabel('Coordinate Value')
    ax4.set_ylabel('Density')
    ax4.legend()
    
    # 5. XYZ Separate Distribution (Boxplot)
    ax5 = fig.add_subplot(gs[2, 0])
    
    if len(result_a.joints._sample_joints) > 0 and len(result_b.joints._sample_joints) > 0:
        data_a = [result_a.joints._sample_joints[:, 0], result_a.joints._sample_joints[:, 1], result_a.joints._sample_joints[:, 2]]
        data_b = [result_b.joints._sample_joints[:, 0], result_b.joints._sample_joints[:, 1], result_b.joints._sample_joints[:, 2]]
        
        positions_a = np.array([1, 2, 3])
        bp_a = ax5.boxplot(data_a, positions=positions_a, widths=0.3, patch_artist=True, boxprops=dict(facecolor="lightblue"))
        
        positions_b = np.array([1, 2, 3]) + 0.4
        bp_b = ax5.boxplot(data_b, positions=positions_b, widths=0.3, patch_artist=True, boxprops=dict(facecolor="moccasin"))
        
        ax5.set_xticks([1.2, 2.2, 3.2])
        ax5.set_xticklabels(['X', 'Y', 'Z'])
        ax5.set_title('XYZ Coordinate Distribution')
        ax5.legend([bp_a["boxes"][0], bp_b["boxes"][0]], [result_a.name, result_b.name])

    # 6. PCA Diversity
    ax6 = fig.add_subplot(gs[2, 1])
    n_comp = min(len(result_a.diversity.pose_pca_explained_variance), 
                 len(result_b.diversity.pose_pca_explained_variance))
    x = np.arange(n_comp)
    ax6.plot(x, np.cumsum(result_a.diversity.pose_pca_explained_variance[:n_comp]), 'o-', label=result_a.name)
    ax6.plot(x, np.cumsum(result_b.diversity.pose_pca_explained_variance[:n_comp]), 's-', label=result_b.name)
    ax6.set_xlabel('PCA Component')
    ax6.set_title('Pose Diversity (Cumulative Variance)')
    ax6.legend()
    
    # 7. Axis Extents (Max - Min)
    ax7 = fig.add_subplot(gs[2, 2])
    range_a = result_a.joints.coord_max - result_a.joints.coord_min
    range_b = result_b.joints.coord_max - result_b.joints.coord_min
    
    x = np.arange(3)
    ax7.bar(x - width/2, range_a, width, label=result_a.name, alpha=0.8)
    ax7.bar(x + width/2, range_b, width, label=result_b.name, alpha=0.8)
    ax7.set_xticks(x)
    ax7.set_xticklabels(['X Range', 'Y Range', 'Z Range'])
    ax7.set_title('Spatial Extent (Max - Min)')
    ax7.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved visualization to {output_dir}/dataset_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='Audit and compare HMR training datasets')
    parser.add_argument('--amass-path', type=str, 
                        default='/path/to/amass',
                        help='Path to AMASS preprocessed data directory')
    parser.add_argument('--nymeria-path', type=str,
                        default='/path/to/nymeria',
                        help='Path to Nymeria preprocessed data directory')
    parser.add_argument('--kind', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to audit')
    parser.add_argument('--output', type=str, default='./audit_results',
                        help='Output directory for reports and visualizations')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Audit both datasets
    amass_result = audit_dataset(args.amass_path, "AMASS", args.kind, is_nymeria=False)
    nymeria_result = audit_dataset(args.nymeria_path, "Nymeria", args.kind, is_nymeria=True)
    
    # Generate comparison report
    report = compare_datasets(amass_result, nymeria_result)
    print(report)
    
    # Save report
    report_path = os.path.join(args.output, 'comparison_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")
    
    # Generate visualizations
    visualize_comparison(amass_result, nymeria_result, args.output)
    
    print("\n✅ Audit complete!")


if __name__ == '__main__':
    main()
