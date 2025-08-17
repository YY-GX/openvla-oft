#!/usr/bin/env python3
"""
Analyze Contact Timesteps Statistics

This script analyzes the contact_timesteps.json file and prints useful statistics
about the contact detection results across all tasks and demonstrations.
"""

import json
import os
import numpy as np
from collections import defaultdict
from pathlib import Path

def load_contact_timesteps(file_path):
    """Load contact timesteps from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_contact_timesteps(contact_data):
    """Analyze contact timesteps and return statistics."""
    
    # Basic statistics
    total_tasks = len(contact_data)
    total_demos = sum(len(task_demos) for task_demos in contact_data.values())
    
    # Per-task statistics
    task_stats = {}
    all_timesteps = []
    
    for task_name, demos in contact_data.items():
        timesteps = list(demos.values())
        task_stats[task_name] = {
            'num_demos': len(demos),
            'min_timestep': min(timesteps) if timesteps else 0,
            'max_timestep': max(timesteps) if timesteps else 0,
            'mean_timestep': np.mean(timesteps) if timesteps else 0,
            'std_timestep': np.std(timesteps) if timesteps else 0,
            'median_timestep': np.median(timesteps) if timesteps else 0
        }
        all_timesteps.extend(timesteps)
    
    # Overall statistics
    overall_stats = {
        'total_tasks': total_tasks,
        'total_demos': total_demos,
        'avg_demos_per_task': total_demos / total_tasks if total_tasks > 0 else 0,
        'min_timestep': min(all_timesteps) if all_timesteps else 0,
        'max_timestep': max(all_timesteps) if all_timesteps else 0,
        'mean_timestep': np.mean(all_timesteps) if all_timesteps else 0,
        'std_timestep': np.std(all_timesteps) if all_timesteps else 0,
        'median_timestep': np.median(all_timesteps) if all_timesteps else 0
    }
    
    return overall_stats, task_stats

def print_statistics(overall_stats, task_stats, contact_data):
    """Print formatted statistics."""
    
    print("=" * 80)
    print("CONTACT TIMESTEPS ANALYSIS")
    print("=" * 80)
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total Tasks: {overall_stats['total_tasks']}")
    print(f"   Total Demos: {overall_stats['total_demos']}")
    print(f"   Avg Demos per Task: {overall_stats['avg_demos_per_task']:.1f}")
    print(f"   Contact Timestep Range: {overall_stats['min_timestep']} - {overall_stats['max_timestep']}")
    print(f"   Mean Contact Timestep: {overall_stats['mean_timestep']:.1f}")
    print(f"   Median Contact Timestep: {overall_stats['median_timestep']:.1f}")
    print(f"   Std Contact Timestep: {overall_stats['std_timestep']:.1f}")
    
    # Task-by-task breakdown
    print(f"\n📋 TASK-BY-TASK BREAKDOWN:")
    print(f"{'Task Name':<50} {'Demos':<8} {'Min':<6} {'Max':<6} {'Mean':<8} {'Std':<6}")
    print("-" * 90)
    
    # Sort tasks by number of demos (descending)
    sorted_tasks = sorted(task_stats.items(), key=lambda x: x[1]['num_demos'], reverse=True)
    
    for task_name, stats in sorted_tasks:
        task_short = task_name[:47] + "..." if len(task_name) > 50 else task_name
        print(f"{task_short:<50} {stats['num_demos']:<8} {stats['min_timestep']:<6} "
              f"{stats['max_timestep']:<6} {stats['mean_timestep']:<8.1f} {stats['std_timestep']:<6.1f}")
    
    # Distribution analysis
    print(f"\n📈 DISTRIBUTION ANALYSIS:")
    all_timesteps = []
    for task_name, demos in contact_data.items():
        all_timesteps.extend(list(demos.values()))
    
    if all_timesteps:
        percentiles = [10, 25, 50, 75, 90]
        print(f"   Percentiles of contact timesteps:")
        for p in percentiles:
            value = np.percentile(all_timesteps, p)
            print(f"     {p}th percentile: {value:.1f}")
    
    # Quality indicators
    print(f"\n✅ QUALITY INDICATORS:")
    tasks_with_demos = sum(1 for stats in task_stats.values() if stats['num_demos'] > 0)
    print(f"   Tasks with contact detections: {tasks_with_demos}/{overall_stats['total_tasks']}")
    
    # Check for potential issues
    early_contacts = sum(1 for stats in task_stats.values() if stats['min_timestep'] < 10)
    late_contacts = sum(1 for stats in task_stats.values() if stats['max_timestep'] > 200)
    
    print(f"   Tasks with very early contacts (<10): {early_contacts}")
    print(f"   Tasks with very late contacts (>200): {late_contacts}")
    
    print("\n" + "=" * 80)

def main():
    """Main function to analyze contact timesteps."""
    
    # Default path to contact timesteps file
    contact_file = "vlm-pose/data_preparation/v2_pipeline/contact_timesteps.json"
    
    # Check if file exists
    if not os.path.exists(contact_file):
        print(f"❌ Error: Contact timesteps file not found at {contact_file}")
        print("Please run the contact detection script first.")
        return
    
    try:
        # Load and analyze data
        contact_data = load_contact_timesteps(contact_file)
        overall_stats, task_stats = analyze_contact_timesteps(contact_data)
        
        # Print statistics
        print_statistics(overall_stats, task_stats, contact_data)
        
        # Save detailed report
        report_file = "contact_timesteps_analysis.txt"
        with open(report_file, 'w') as f:
            f.write("CONTACT TIMESTEPS ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Statistics:\n")
            for key, value in overall_stats.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nTask Statistics:\n")
            for task_name, stats in task_stats.items():
                f.write(f"  {task_name}: {stats}\n")
        
        print(f"📄 Detailed report saved to: {report_file}")
        
    except Exception as e:
        print(f"❌ Error analyzing contact timesteps: {str(e)}")

if __name__ == "__main__":
    main() 