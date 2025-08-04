import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import json

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib or seaborn not available. Visualization will be skipped.")

def analyze_pose_statistics(poses, annotation_df=None):
    """
    Analyze pose statistics to check if poses are normalized.
    
    Args:
        poses: numpy array of shape (N, 6) where each pose is [x, y, z, rx, ry, rz]
        annotation_df: optional pandas DataFrame with annotation info
    
    Returns:
        dict: statistics for position and orientation components
    """
    print(f"Analyzing {len(poses)} poses...")
    
    # Split into position and orientation
    positions = poses[:, :3]  # x, y, z
    orientations = poses[:, 3:]  # rx, ry, rz (axis-angle)
    
    # Position statistics
    pos_stats = {
        'x': {
            'min': float(np.min(positions[:, 0])),
            'max': float(np.max(positions[:, 0])),
            'mean': float(np.mean(positions[:, 0])),
            'std': float(np.std(positions[:, 0])),
            'var': float(np.var(positions[:, 0])),
            'range': float(np.max(positions[:, 0]) - np.min(positions[:, 0]))
        },
        'y': {
            'min': float(np.min(positions[:, 1])),
            'max': float(np.max(positions[:, 1])),
            'mean': float(np.mean(positions[:, 1])),
            'std': float(np.std(positions[:, 1])),
            'var': float(np.var(positions[:, 1])),
            'range': float(np.max(positions[:, 1]) - np.min(positions[:, 1]))
        },
        'z': {
            'min': float(np.min(positions[:, 2])),
            'max': float(np.max(positions[:, 2])),
            'mean': float(np.mean(positions[:, 2])),
            'std': float(np.std(positions[:, 2])),
            'var': float(np.var(positions[:, 2])),
            'range': float(np.max(positions[:, 2]) - np.min(positions[:, 2]))
        }
    }
    
    # Orientation statistics (axis-angle)
    orient_stats = {
        'rx': {
            'min': float(np.min(orientations[:, 0])),
            'max': float(np.max(orientations[:, 0])),
            'mean': float(np.mean(orientations[:, 0])),
            'std': float(np.std(orientations[:, 0])),
            'var': float(np.var(orientations[:, 0])),
            'range': float(np.max(orientations[:, 0]) - np.min(orientations[:, 0]))
        },
        'ry': {
            'min': float(np.min(orientations[:, 1])),
            'max': float(np.max(orientations[:, 1])),
            'mean': float(np.mean(orientations[:, 1])),
            'std': float(np.std(orientations[:, 1])),
            'var': float(np.var(orientations[:, 1])),
            'range': float(np.max(orientations[:, 1]) - np.min(orientations[:, 1]))
        },
        'rz': {
            'min': float(np.min(orientations[:, 2])),
            'max': float(np.max(orientations[:, 2])),
            'mean': float(np.mean(orientations[:, 2])),
            'std': float(np.std(orientations[:, 2])),
            'var': float(np.var(orientations[:, 2])),
            'range': float(np.max(orientations[:, 2]) - np.min(orientations[:, 2]))
        }
    }
    
    # Overall statistics
    overall_stats = {
        'total_poses': len(poses),
        'position_range': {
            'x_range': pos_stats['x']['range'],
            'y_range': pos_stats['y']['range'],
            'z_range': pos_stats['z']['range'],
            'max_range': max(pos_stats['x']['range'], pos_stats['y']['range'], pos_stats['z']['range'])
        },
        'orientation_range': {
            'rx_range': orient_stats['rx']['range'],
            'ry_range': orient_stats['ry']['range'],
            'rz_range': orient_stats['rz']['range'],
            'max_range': max(orient_stats['rx']['range'], orient_stats['ry']['range'], orient_stats['rz']['range'])
        }
    }
    
    # Check for potential normalization indicators
    normalization_indicators = {
        'position_normalized': (
            pos_stats['x']['range'] < 2.0 and 
            pos_stats['y']['range'] < 2.0 and 
            pos_stats['z']['range'] < 2.0
        ),
        'orientation_normalized': (
            orient_stats['rx']['range'] < 2*np.pi and 
            orient_stats['ry']['range'] < 2*np.pi and 
            orient_stats['rz']['range'] < 2*np.pi
        ),
        'likely_normalized': (
            pos_stats['x']['std'] < 1.0 and 
            pos_stats['y']['std'] < 1.0 and 
            pos_stats['z']['std'] < 1.0 and
            orient_stats['rx']['std'] < 1.0 and 
            orient_stats['ry']['std'] < 1.0 and 
            orient_stats['rz']['std'] < 1.0
        )
    }
    
    return {
        'position_stats': pos_stats,
        'orientation_stats': orient_stats,
        'overall_stats': overall_stats,
        'normalization_indicators': normalization_indicators
    }

def print_statistics(stats):
    """Print formatted statistics."""
    print("\n" + "="*60)
    print("POSE STATISTICS ANALYSIS")
    print("="*60)
    
    print(f"\nTotal poses analyzed: {stats['overall_stats']['total_poses']}")
    
    print("\n" + "-"*40)
    print("POSITION STATISTICS (x, y, z)")
    print("-"*40)
    for axis in ['x', 'y', 'z']:
        s = stats['position_stats'][axis]
        print(f"{axis.upper()}: min={s['min']:.4f}, max={s['max']:.4f}, "
              f"mean={s['mean']:.4f}, std={s['std']:.4f}, range={s['range']:.4f}")
    
    print("\n" + "-"*40)
    print("ORIENTATION STATISTICS (rx, ry, rz - axis-angle)")
    print("-"*40)
    for axis in ['rx', 'ry', 'rz']:
        s = stats['orientation_stats'][axis]
        print(f"{axis.upper()}: min={s['min']:.4f}, max={s['max']:.4f}, "
              f"mean={s['mean']:.4f}, std={s['std']:.4f}, range={s['range']:.4f}")
    
    print("\n" + "-"*40)
    print("NORMALIZATION ANALYSIS")
    print("-"*40)
    indicators = stats['normalization_indicators']
    print(f"Position ranges < 2.0: {indicators['position_normalized']}")
    print(f"Orientation ranges < 2π: {indicators['orientation_normalized']}")
    print(f"Standard deviations < 1.0: {indicators['likely_normalized']}")
    
    if indicators['likely_normalized']:
        print("✓ Poses appear to be normalized (std < 1.0)")
    elif indicators['position_normalized'] and indicators['orientation_normalized']:
        print("⚠ Poses may be partially normalized (ranges are reasonable)")
    else:
        print("✗ Poses do not appear to be normalized")

def create_visualizations(poses, save_dir):
    """Create histograms and scatter plots of pose distributions."""
    if not PLOTTING_AVAILABLE:
        print("Skipping visualization - matplotlib/seaborn not available")
        return
        
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    positions = poses[:, :3]
    orientations = poses[:, 3:]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Pose Distribution Analysis', fontsize=16)
    
    # Position histograms
    axes[0, 0].hist(positions[:, 0], bins=50, alpha=0.7, color='red')
    axes[0, 0].set_title('X Position Distribution')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Frequency')
    
    axes[0, 1].hist(positions[:, 1], bins=50, alpha=0.7, color='green')
    axes[0, 1].set_title('Y Position Distribution')
    axes[0, 1].set_xlabel('Y')
    axes[0, 1].set_ylabel('Frequency')
    
    axes[0, 2].hist(positions[:, 2], bins=50, alpha=0.7, color='blue')
    axes[0, 2].set_title('Z Position Distribution')
    axes[0, 2].set_xlabel('Z')
    axes[0, 2].set_ylabel('Frequency')
    
    # Orientation histograms
    axes[1, 0].hist(orientations[:, 0], bins=50, alpha=0.7, color='orange')
    axes[1, 0].set_title('RX Orientation Distribution')
    axes[1, 0].set_xlabel('RX (axis-angle)')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].hist(orientations[:, 1], bins=50, alpha=0.7, color='purple')
    axes[1, 1].set_title('RY Orientation Distribution')
    axes[1, 1].set_xlabel('RY (axis-angle)')
    axes[1, 1].set_ylabel('Frequency')
    
    axes[1, 2].hist(orientations[:, 2], bins=50, alpha=0.7, color='brown')
    axes[1, 2].set_title('RZ Orientation Distribution')
    axes[1, 2].set_xlabel('RZ (axis-angle)')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'pose_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    pose_df = pd.DataFrame(poses, columns=['x', 'y', 'z', 'rx', 'ry', 'rz'])
    correlation_matrix = pose_df.corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=ax)
    ax.set_title('Pose Component Correlations')
    plt.tight_layout()
    plt.savefig(save_dir / 'pose_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {save_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze pose statistics in dataset")
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of dataset')
    parser.add_argument('--split', type=str, default='val', help='Dataset split (train/val/test)')
    parser.add_argument('--splits_folder', type=str, default='smaller_splits', help='Folder name for annotation splits')
    parser.add_argument('--save_dir', type=str, default='runs/eval_results/pose_analysis', help='Directory to save results')
    parser.add_argument('--create_plots', action='store_true', help='Create visualization plots')
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    split = args.split
    splits_folder = args.splits_folder
    save_dir = Path(args.save_dir)
    
    # Load poses
    poses_path = data_root / "local_poses.npy"
    if not poses_path.exists():
        raise FileNotFoundError(f"Poses file not found: {poses_path}")
    
    poses = np.load(poses_path)
    print(f"Loaded poses from {poses_path}")
    print(f"Pose array shape: {poses.shape}")
    
    # Load annotation if available
    annotation_df = None
    annotation_path = data_root / f"{split}_annotation.csv"
    if not annotation_path.exists():
        annotation_path = data_root / splits_folder / f"{split}_annotation.csv"
    
    if annotation_path.exists():
        annotation_df = pd.read_csv(annotation_path)
        print(f"Loaded annotation from {annotation_path}")
        print(f"Annotation shape: {annotation_df.shape}")
        
        # Filter poses by split if annotation has split info
        if 'split' in annotation_df.columns:
            split_mask = annotation_df['split'] == split
            poses = poses[split_mask]
            print(f"Filtered to {len(poses)} poses for {split} split")
    
    # Analyze statistics
    stats = analyze_pose_statistics(poses, annotation_df)
    print_statistics(stats)
    
    # Save results
    save_dir.mkdir(parents=True, exist_ok=True)
    stats_path = save_dir / f"{split}_pose_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to {stats_path}")
    
    # Create visualizations if requested
    if args.create_plots:
        create_visualizations(poses, save_dir)

if __name__ == "__main__":
    main() 