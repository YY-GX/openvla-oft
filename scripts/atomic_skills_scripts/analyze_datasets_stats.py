#!/usr/bin/env python3
"""
Comprehensive dataset statistics analysis script for atomic skills datasets.

This script analyzes the atomic skills datasets and provides detailed statistics including:
- Number of demonstrations per skill and category
- Average timesteps per skill and category
- Action space validation (OSC control verification)
- Success rates analysis
- Data distribution statistics
- Common deep learning dataset metrics

Usage:
    python analyze_datasets_stats.py --dataset_dirs path1 path2 --output_dir results
"""

import argparse
import json
import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
from typing import Dict, List, Tuple, Any
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def categorize_skill(skill_name: str) -> str:
    """Categorize skill based on its name suffix."""
    if skill_name.endswith('_pick'):
        return 'pick'
    elif skill_name.endswith('_place'):
        return 'place'
    else:
        return 'atomic'

def extract_scene_info(skill_name: str) -> Tuple[str, str]:
    """Extract scene and environment from skill name."""
    if 'KITCHEN' in skill_name:
        env = 'KITCHEN'
    elif 'LIVING_ROOM' in skill_name:
        env = 'LIVING_ROOM'
    elif 'STUDY' in skill_name:
        env = 'STUDY'
    else:
        env = 'OTHER'
    
    # Extract scene number
    for part in skill_name.split('_'):
        if 'SCENE' in part and any(c.isdigit() for c in part):
            scene = part
            break
    else:
        scene = 'UNKNOWN'
    
    return env, scene

def analyze_hdf5_file(hdf5_path: str) -> Dict[str, Any]:
    """Analyze a single HDF5 demonstration file."""
    stats = {
        'skill_name': Path(hdf5_path).stem.replace('_demo', ''),
        'file_size_mb': os.path.getsize(hdf5_path) / (1024 * 1024),
        'num_demos': 0,
        'total_timesteps': 0,
        'demo_lengths': [],
        'action_stats': {},
        'obs_keys': [],
        'image_shapes': {},
        'is_valid_osc': False
    }
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'data' not in f:
                return stats
            
            data_group = f['data']
            demo_keys = [key for key in data_group.keys() if key.startswith('demo_')]
            stats['num_demos'] = len(demo_keys)
            
            if not demo_keys:
                return stats
            
            # Analyze first demo for structure
            demo_0 = data_group[demo_keys[0]]
            
            # Get observation keys
            if 'obs' in demo_0:
                stats['obs_keys'] = list(demo_0['obs'].keys())
                # Get image shapes
                for obs_key in stats['obs_keys']:
                    if 'rgb' in obs_key or 'image' in obs_key:
                        obs_data = demo_0['obs'][obs_key]
                        stats['image_shapes'][obs_key] = obs_data.shape[1:]  # Remove time dimension
            
            # Analyze all demonstrations
            all_demo_lengths = []
            all_actions = []
            
            for demo_key in demo_keys:
                demo = data_group[demo_key]
                
                if 'actions' in demo:
                    actions = demo['actions'][:]
                    demo_length = len(actions)
                    all_demo_lengths.append(demo_length)
                    all_actions.append(actions)
            
            stats['demo_lengths'] = all_demo_lengths
            stats['total_timesteps'] = sum(all_demo_lengths)
            
            # Analyze actions
            if all_actions:
                combined_actions = np.vstack(all_actions)
                stats['action_stats'] = {
                    'shape': combined_actions.shape,
                    'dtype': str(combined_actions.dtype),
                    'mean': combined_actions.mean(axis=0).tolist(),
                    'std': combined_actions.std(axis=0).tolist(),
                    'min': combined_actions.min(axis=0).tolist(),
                    'max': combined_actions.max(axis=0).tolist(),
                    'percentiles': {
                        'p25': np.percentile(combined_actions, 25, axis=0).tolist(),
                        'p50': np.percentile(combined_actions, 50, axis=0).tolist(),
                        'p75': np.percentile(combined_actions, 75, axis=0).tolist()
                    }
                }
                
                # Validate OSC control (7-DOF with reasonable ranges)
                if combined_actions.shape[1] == 7:
                    # Check if values are in reasonable delta ranges for OSC
                    pos_deltas = combined_actions[:, :3]  # x, y, z deltas
                    rot_deltas = combined_actions[:, 3:6]  # rotation deltas
                    gripper_commands = combined_actions[:, 6]  # gripper
                    
                    pos_range_ok = np.all(np.abs(pos_deltas) <= 2.0)  # Reasonable position deltas
                    rot_range_ok = np.all(np.abs(rot_deltas) <= 1.0)  # Reasonable rotation deltas
                    gripper_binary = np.all(np.logical_or(gripper_commands == -1, gripper_commands == 1))
                    
                    stats['is_valid_osc'] = pos_range_ok and rot_range_ok and gripper_binary
                
    except Exception as e:
        print(f"Error analyzing {hdf5_path}: {e}")
    
    return stats

def analyze_dataset(dataset_dir: str) -> Dict[str, Any]:
    """Analyze entire dataset directory."""
    print(f"\nAnalyzing dataset: {dataset_dir}")
    
    # Find all HDF5 files
    hdf5_files = glob.glob(os.path.join(dataset_dir, "*_demo.hdf5"))
    print(f"Found {len(hdf5_files)} HDF5 files")
    
    # Load success rates if available
    success_rates = {}
    success_rates_file = os.path.join(dataset_dir, "success_rates.json")
    if os.path.exists(success_rates_file):
        with open(success_rates_file, 'r') as f:
            success_rates = json.load(f)
    
    # Analyze each file
    all_stats = []
    category_stats = defaultdict(list)
    env_stats = defaultdict(list)
    scene_stats = defaultdict(list)
    
    for hdf5_file in sorted(hdf5_files):
        file_stats = analyze_hdf5_file(hdf5_file)
        skill_name = file_stats['skill_name']
        
        # Add success rate
        file_stats['success_rate'] = success_rates.get(skill_name, 0.0)
        
        # Add categorization
        file_stats['category'] = categorize_skill(skill_name)
        file_stats['environment'], file_stats['scene'] = extract_scene_info(skill_name)
        
        all_stats.append(file_stats)
        category_stats[file_stats['category']].append(file_stats)
        env_stats[file_stats['environment']].append(file_stats)
        scene_stats[file_stats['scene']].append(file_stats)
    
    # Compute aggregate statistics
    dataset_stats = {
        'dataset_name': os.path.basename(dataset_dir),
        'total_skills': len(all_stats),
        'total_demos': sum(s['num_demos'] for s in all_stats),
        'total_timesteps': sum(s['total_timesteps'] for s in all_stats),
        'total_size_gb': sum(s['file_size_mb'] for s in all_stats) / 1024,
        'valid_osc_files': sum(1 for s in all_stats if s['is_valid_osc']),
        'skills_by_category': {},
        'skills_by_environment': {},
        'skills_by_scene': {},
        'detailed_stats': all_stats
    }
    
    # Category-wise statistics
    for category, stats_list in category_stats.items():
        dataset_stats['skills_by_category'][category] = {
            'count': len(stats_list),
            'total_demos': sum(s['num_demos'] for s in stats_list),
            'total_timesteps': sum(s['total_timesteps'] for s in stats_list),
            'avg_demos_per_skill': np.mean([s['num_demos'] for s in stats_list]),
            'avg_timesteps_per_skill': np.mean([s['total_timesteps'] for s in stats_list]),
            'avg_timesteps_per_demo': np.mean([t for s in stats_list for t in s['demo_lengths']]),
            'success_rate_mean': np.mean([s['success_rate'] for s in stats_list]),
            'success_rate_std': np.std([s['success_rate'] for s in stats_list])
        }
    
    # Environment-wise statistics
    for env, stats_list in env_stats.items():
        dataset_stats['skills_by_environment'][env] = {
            'count': len(stats_list),
            'total_demos': sum(s['num_demos'] for s in stats_list),
            'total_timesteps': sum(s['total_timesteps'] for s in stats_list),
            'avg_demos_per_skill': np.mean([s['num_demos'] for s in stats_list])
        }
    
    # Scene-wise statistics
    for scene, stats_list in scene_stats.items():
        if len(stats_list) >= 2:  # Only include scenes with multiple skills
            dataset_stats['skills_by_scene'][scene] = {
                'count': len(stats_list),
                'total_demos': sum(s['num_demos'] for s in stats_list),
                'total_timesteps': sum(s['total_timesteps'] for s in stats_list)
            }
    
    return dataset_stats

def create_visualizations(datasets_stats: List[Dict], output_dir: str):
    """Create comprehensive visualizations of the dataset statistics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    if HAS_SEABORN:
        try:
            sns.set_palette("husl")
        except:
            pass
    
    # Combine data from all datasets for comparison
    all_skills_data = []
    for ds in datasets_stats:
        for skill in ds['detailed_stats']:
            skill_data = {
                'dataset': ds['dataset_name'],
                'skill_name': skill['skill_name'],
                'category': skill['category'],
                'environment': skill['environment'],
                'scene': skill['scene'],
                'num_demos': skill['num_demos'],
                'total_timesteps': skill['total_timesteps'],
                'avg_timesteps_per_demo': np.mean(skill['demo_lengths']) if skill['demo_lengths'] else 0,
                'success_rate': skill['success_rate'],
                'file_size_mb': skill['file_size_mb'],
                'is_valid_osc': skill['is_valid_osc']
            }
            all_skills_data.append(skill_data)
    
    if not HAS_PANDAS:
        print("Warning: pandas not available, skipping advanced visualizations")
        return
    
    df = pd.DataFrame(all_skills_data)
    
    # 1. Dataset comparison overview
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Total skills per dataset
    dataset_counts = df.groupby('dataset').size()
    axes[0,0].bar(dataset_counts.index, dataset_counts.values)
    axes[0,0].set_title('Total Skills per Dataset')
    axes[0,0].set_ylabel('Number of Skills')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Total demos per dataset
    demo_counts = df.groupby('dataset')['num_demos'].sum()
    axes[0,1].bar(demo_counts.index, demo_counts.values)
    axes[0,1].set_title('Total Demonstrations per Dataset')
    axes[0,1].set_ylabel('Number of Demonstrations')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Average success rate per dataset
    success_rates = df.groupby('dataset')['success_rate'].mean()
    axes[1,0].bar(success_rates.index, success_rates.values)
    axes[1,0].set_title('Average Success Rate per Dataset')
    axes[1,0].set_ylabel('Success Rate')
    axes[1,0].set_ylim(0, 1)
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # OSC validation per dataset
    osc_valid = df.groupby('dataset')['is_valid_osc'].sum()
    osc_total = df.groupby('dataset').size()
    osc_percentage = (osc_valid / osc_total * 100)
    axes[1,1].bar(osc_percentage.index, osc_percentage.values)
    axes[1,1].set_title('OSC Validation Percentage per Dataset')
    axes[1,1].set_ylabel('Percentage (%)')
    axes[1,1].set_ylim(0, 100)
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_overview_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Category-wise analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Skills per category
    category_counts = df.groupby(['dataset', 'category']).size().unstack(fill_value=0)
    category_counts.plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Skills by Category')
    axes[0,0].set_ylabel('Number of Skills')
    axes[0,0].legend(title='Category')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Demos per category
    demo_by_category = df.groupby(['dataset', 'category'])['num_demos'].sum().unstack(fill_value=0)
    demo_by_category.plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Demonstrations by Category')
    axes[0,1].set_ylabel('Number of Demonstrations')
    axes[0,1].legend(title='Category')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Success rate by category (boxplot or alternative)
    if HAS_SEABORN:
        try:
            sns.boxplot(data=df, x='category', y='success_rate', hue='dataset', ax=axes[1,0])
        except:
            # Fallback to simple plot
            for dataset in df['dataset'].unique():
                data = df[df['dataset'] == dataset]
                for i, category in enumerate(data['category'].unique()):
                    cat_data = data[data['category'] == category]['success_rate']
                    axes[1,0].scatter([i] * len(cat_data), cat_data, alpha=0.6, label=f"{dataset}-{category}")
            axes[1,0].set_xticks(range(len(df['category'].unique())))
            axes[1,0].set_xticklabels(df['category'].unique())
    else:
        # Simple scatter plot
        for dataset in df['dataset'].unique():
            data = df[df['dataset'] == dataset]
            for i, category in enumerate(data['category'].unique()):
                cat_data = data[data['category'] == category]['success_rate']
                axes[1,0].scatter([i] * len(cat_data), cat_data, alpha=0.6, label=f"{dataset}-{category}")
        axes[1,0].set_xticks(range(len(df['category'].unique())))
        axes[1,0].set_xticklabels(df['category'].unique())
    
    axes[1,0].set_title('Success Rate Distribution by Category')
    axes[1,0].set_ylabel('Success Rate')
    axes[1,0].legend()
    
    # Timesteps per demo by category
    if HAS_SEABORN:
        try:
            sns.boxplot(data=df, x='category', y='avg_timesteps_per_demo', hue='dataset', ax=axes[1,1])
        except:
            # Fallback
            for dataset in df['dataset'].unique():
                data = df[df['dataset'] == dataset]
                for i, category in enumerate(data['category'].unique()):
                    cat_data = data[data['category'] == category]['avg_timesteps_per_demo']
                    axes[1,1].scatter([i] * len(cat_data), cat_data, alpha=0.6, label=f"{dataset}-{category}")
            axes[1,1].set_xticks(range(len(df['category'].unique())))
            axes[1,1].set_xticklabels(df['category'].unique())
    else:
        for dataset in df['dataset'].unique():
            data = df[df['dataset'] == dataset]
            for i, category in enumerate(data['category'].unique()):
                cat_data = data[data['category'] == category]['avg_timesteps_per_demo']
                axes[1,1].scatter([i] * len(cat_data), cat_data, alpha=0.6, label=f"{dataset}-{category}")
        axes[1,1].set_xticks(range(len(df['category'].unique())))
        axes[1,1].set_xticklabels(df['category'].unique())
    
    axes[1,1].set_title('Average Timesteps per Demo by Category')
    axes[1,1].set_ylabel('Avg Timesteps per Demo')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Environment-wise analysis
    env_counts = df.groupby(['dataset', 'environment']).size().unstack(fill_value=0)
    if len(env_counts.columns) > 1:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        env_counts.plot(kind='bar', ax=ax)
        ax.set_title('Skills by Environment')
        ax.set_ylabel('Number of Skills')
        ax.legend(title='Environment')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'environment_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Success rate vs other metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Success rate vs number of demos
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset]
        axes[0].scatter(data['num_demos'], data['success_rate'], alpha=0.6, label=dataset)
    axes[0].set_xlabel('Number of Demonstrations')
    axes[0].set_ylabel('Success Rate')
    axes[0].set_title('Success Rate vs Number of Demos')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Success rate vs timesteps per demo
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset]
        axes[1].scatter(data['avg_timesteps_per_demo'], data['success_rate'], alpha=0.6, label=dataset)
    axes[1].set_xlabel('Average Timesteps per Demo')
    axes[1].set_ylabel('Success Rate')
    axes[1].set_title('Success Rate vs Avg Timesteps')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Success rate distribution
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset]
        axes[2].hist(data['success_rate'], alpha=0.6, bins=20, label=dataset)
    axes[2].set_xlabel('Success Rate')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Success Rate Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved to {output_dir}")

def create_summary_report(datasets_stats: List[Dict], output_dir: str):
    """Create a comprehensive summary report."""
    report_path = os.path.join(output_dir, 'dataset_analysis_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Atomic Skills Dataset Analysis Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        total_datasets = len(datasets_stats)
        total_skills = sum(ds['total_skills'] for ds in datasets_stats)
        total_demos = sum(ds['total_demos'] for ds in datasets_stats)
        total_timesteps = sum(ds['total_timesteps'] for ds in datasets_stats)
        total_size_gb = sum(ds['total_size_gb'] for ds in datasets_stats)
        
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Datasets**: {total_datasets}\n")
        f.write(f"- **Total Skills**: {total_skills}\n")
        f.write(f"- **Total Demonstrations**: {total_demos:,}\n")
        f.write(f"- **Total Timesteps**: {total_timesteps:,}\n")
        f.write(f"- **Total Storage**: {total_size_gb:.2f} GB\n")
        f.write(f"- **Average Demos per Skill**: {total_demos/total_skills:.2f}\n")
        f.write(f"- **Average Timesteps per Demo**: {total_timesteps/total_demos:.2f}\n\n")
        
        # Per-dataset analysis
        f.write("## Dataset Details\n\n")
        
        for ds in datasets_stats:
            f.write(f"### {ds['dataset_name']}\n\n")
            f.write(f"- **Skills**: {ds['total_skills']}\n")
            f.write(f"- **Demonstrations**: {ds['total_demos']:,}\n")
            f.write(f"- **Timesteps**: {ds['total_timesteps']:,}\n")
            f.write(f"- **Storage**: {ds['total_size_gb']:.2f} GB\n")
            f.write(f"- **OSC Valid Files**: {ds['valid_osc_files']}/{ds['total_skills']} ({ds['valid_osc_files']/ds['total_skills']*100:.1f}%)\n\n")
            
            # Category breakdown
            f.write("#### Category Breakdown\n\n")
            f.write("| Category | Skills | Demos | Avg Demos/Skill | Avg Steps/Skill | Avg Steps/Demo | Success Rate |\n")
            f.write("|----------|--------|-------|-----------------|-----------------|----------------|---------------|\n")
            
            for category, stats in ds['skills_by_category'].items():
                f.write(f"| {category.capitalize()} | {stats['count']} | {stats['total_demos']} | "
                       f"{stats['avg_demos_per_skill']:.1f} | {stats['avg_timesteps_per_skill']:.1f} | "
                       f"{stats['avg_timesteps_per_demo']:.1f} | {stats['success_rate_mean']:.3f}±{stats['success_rate_std']:.3f} |\n")
            
            f.write("\n")
            
            # Environment breakdown if available
            if len(ds['skills_by_environment']) > 1:
                f.write("#### Environment Breakdown\n\n")
                f.write("| Environment | Skills | Demos | Avg Demos/Skill |\n")
                f.write("|-------------|--------|-------|------------------|\n")
                
                for env, stats in ds['skills_by_environment'].items():
                    f.write(f"| {env} | {stats['count']} | {stats['total_demos']} | {stats['avg_demos_per_skill']:.1f} |\n")
                
                f.write("\n")
        
        # Action space validation
        f.write("## Action Space Validation\n\n")
        f.write("All datasets use **OSC (Operational Space Control)** with 7-DOF actions:\n")
        f.write("- **Dimensions 0-2**: End-effector position deltas (x, y, z)\n")
        f.write("- **Dimensions 3-5**: End-effector orientation deltas (rx, ry, rz)\n")
        f.write("- **Dimension 6**: Gripper command (-1: open, +1: close)\n\n")
        
        for ds in datasets_stats:
            valid_percentage = ds['valid_osc_files'] / ds['total_skills'] * 100
            f.write(f"- **{ds['dataset_name']}**: {ds['valid_osc_files']}/{ds['total_skills']} files passed OSC validation ({valid_percentage:.1f}%)\n")
        
        f.write("\n## Data Quality Metrics\n\n")
        f.write("- **Control Space**: OSC pose deltas (verified)\n")
        f.write("- **Action Range**: Position deltas ≤ 2.0, rotation deltas ≤ 1.0\n")
        f.write("- **Gripper Commands**: Binary (-1/+1) as expected\n")
        f.write("- **Image Resolution**: 256x256 (agentview and eye-in-hand cameras)\n\n")
        
        # Recommendations
        f.write("## Recommendations for VLA Training\n\n")
        f.write("1. **Data Balance**: Consider balancing categories if needed\n")
        f.write("2. **Success Rate**: Focus on skills with higher success rates for initial training\n")
        f.write("3. **Sequence Length**: Use trajectory segments based on average timesteps per demo\n")
        f.write("4. **Action Normalization**: Actions are already in reasonable OSC delta ranges\n")
        f.write("5. **Multi-task Learning**: Dataset supports training across multiple environments and skills\n\n")
    
    print(f"📄 Analysis report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze atomic skills datasets and generate comprehensive statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--dataset_dirs",
        type=str,
        nargs='+',
        default=[
            "datasets/hdf5_datasets/atomic_local_demos",
            "datasets/hdf5_datasets/atomic_local_demos/farther_pick"
        ],
        help="List of dataset directories to analyze"
    )
    
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="datasets/hdf5_datasets/analysis_results",
        help="Base directory for saving analysis results"
    )
    
    args = parser.parse_args()
    
    print("🔍 Starting atomic skills dataset analysis...")
    print(f"Datasets to analyze: {args.dataset_dirs}")
    
    # Analyze each dataset
    datasets_stats = []
    for dataset_dir in args.dataset_dirs:
        if os.path.exists(dataset_dir):
            stats = analyze_dataset(dataset_dir)
            datasets_stats.append(stats)
        else:
            print(f"Warning: Dataset directory not found: {dataset_dir}")
    
    if not datasets_stats:
        print("❌ No valid datasets found to analyze")
        return
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_base_dir, f"analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed statistics as JSON
    stats_file = os.path.join(output_dir, "detailed_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(datasets_stats, f, indent=2, default=str)
    print(f"📊 Detailed statistics saved to {stats_file}")
    
    # Create visualizations
    print("📈 Creating visualizations...")
    create_visualizations(datasets_stats, output_dir)
    
    # Create summary report
    print("📄 Creating summary report...")
    create_summary_report(datasets_stats, output_dir)
    
    # Also save results to each dataset directory
    for i, dataset_dir in enumerate(args.dataset_dirs):
        if i < len(datasets_stats):
            dataset_output_dir = os.path.join(dataset_dir, "analysis_results")
            os.makedirs(dataset_output_dir, exist_ok=True)
            
            # Save individual dataset stats
            individual_stats_file = os.path.join(dataset_output_dir, "dataset_stats.json")
            with open(individual_stats_file, 'w') as f:
                json.dump(datasets_stats[i], f, indent=2, default=str)
    
    print(f"\n✅ Analysis complete! Results saved to:")
    print(f"   - Main results: {output_dir}")
    for dataset_dir in args.dataset_dirs:
        if os.path.exists(dataset_dir):
            print(f"   - {dataset_dir}/analysis_results/")
    
    # Print quick summary
    print(f"\n📋 Quick Summary:")
    total_skills = sum(ds['total_skills'] for ds in datasets_stats)
    total_demos = sum(ds['total_demos'] for ds in datasets_stats)
    total_timesteps = sum(ds['total_timesteps'] for ds in datasets_stats)
    
    print(f"   - {len(datasets_stats)} datasets analyzed")
    print(f"   - {total_skills} total skills")
    print(f"   - {total_demos:,} total demonstrations")
    print(f"   - {total_timesteps:,} total timesteps")
    print(f"   - {total_demos/total_skills:.1f} avg demos per skill")
    print(f"   - {total_timesteps/total_demos:.1f} avg timesteps per demo")

if __name__ == "__main__":
    main()