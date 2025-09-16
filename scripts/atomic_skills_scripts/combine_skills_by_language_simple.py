#!/usr/bin/env python3
"""
Combine Atomic Skills by Language (Simplified Version)

This script combines atomic skills HDF5 files based on their BDDL language descriptions.
Skills with the same language are aggregated into single HDF5 files.

Usage:
    python combine_skills_by_language_simple.py --input_dir datasets/hdf5_datasets/atomic_local_demos/closer_pick --output_dir datasets/hdf5_datasets/atomic_local_demos/closer_pick_combined
"""

import argparse
import os
import glob
import h5py
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List, Tuple, Any


def get_skill_language_from_bddl(bddl_file_path: str) -> str:
    """
    Extract language description from BDDL file.
    
    Args:
        bddl_file_path: Path to the BDDL file
        
    Returns:
        Language description string
    """
    try:
        with open(bddl_file_path, 'r') as f:
            content = f.read()
        
        # Look for (:language ...) line
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('(:language'):
                # Extract the language part after (:language
                language = line.replace('(:language', '').strip().rstrip(')')
                return language
        
        # Fallback: if no language found, use filename
        return os.path.basename(bddl_file_path).replace('.bddl', '')
        
    except Exception as e:
        print(f"Warning: Could not read BDDL file {bddl_file_path}: {e}")
        # Fallback: use filename
        return os.path.basename(bddl_file_path).replace('.bddl', '')


def load_hdf5_file(file_path: str) -> Dict[str, Any]:
    """
    Load HDF5 file into a dictionary.
    
    Args:
        file_path: Path to HDF5 file
        
    Returns:
        Dictionary containing the HDF5 data
    """
    def recursively_extract(group):
        result = {}
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Dataset):
                result[key] = item[()]
            elif isinstance(item, h5py.Group):
                result[key] = recursively_extract(item)
        return result

    with h5py.File(file_path, 'r') as file:
        return recursively_extract(file)


def combine_demos_by_language(input_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    Combine HDF5 demo files by language description.
    
    Args:
        input_dir: Directory containing input HDF5 files
        output_dir: Directory to save combined HDF5 files
        
    Returns:
        Dictionary with combination statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all HDF5 files
    hdf5_files = glob.glob(os.path.join(input_dir, "*.hdf5"))
    if not hdf5_files:
        print(f"No HDF5 files found in {input_dir}")
        return {}
    
    print(f"Found {len(hdf5_files)} HDF5 files to process")
    
    # Group files by language
    language_groups = defaultdict(list)
    
    for hdf5_file in tqdm(hdf5_files, desc="Analyzing files"):
        # Extract skill name from filename
        filename = os.path.basename(hdf5_file)
        skill_name = filename.replace('_demo.hdf5', '')
        
        # Try to find corresponding BDDL file
        bddl_file = os.path.join('externals/boss/libero/libero/bddl_files/atomic_skills', f"{skill_name}.bddl")
        if os.path.exists(bddl_file):
            language = get_skill_language_from_bddl(bddl_file)
        else:
            # Fallback: use filename as language
            language = skill_name
        
        language_groups[language].append(hdf5_file)
        print(f"  {skill_name} -> {language}")
    
    print(f"\nFound {len(language_groups)} unique language descriptions:")
    for language, files in language_groups.items():
        print(f"  '{language}': {len(files)} files")
    
    # Combine demos for each language
    combined_stats = {}
    
    for language, files in tqdm(language_groups.items(), desc="Combining demos"):
        print(f"\nCombining {len(files)} files for language: '{language}'")
        
        # Load all demos for this language
        all_demos = []
        total_demos = 0
        step_counts = []  # Track step counts for statistics
        
        for file_path in files:
            try:
                data = load_hdf5_file(file_path)
                if 'data' in data:
                    file_demos = list(data['data'].keys())
                    total_demos += len(file_demos)
                    
                    # Add file info to each demo and collect step counts
                    for demo_key in file_demos:
                        demo_data = data['data'][demo_key]
                        demo_data['_source_file'] = os.path.basename(file_path)
                        demo_data['_source_demo_key'] = demo_key
                        all_demos.append(demo_data)
                        
                        # Collect step count for this demo
                        if 'actions' in demo_data:
                            step_counts.append(len(demo_data['actions']))
                        
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue
        
        if not all_demos:
            print(f"Warning: No valid demos found for language '{language}'")
            continue
        
        # Calculate step statistics
        step_stats = {}
        if step_counts:
            step_stats = {
                'min_steps': min(step_counts),
                'max_steps': max(step_counts),
                'mean_steps': round(np.mean(step_counts), 2),
                'total_steps': sum(step_counts)
            }
        
        # Create combined HDF5 file with language name joined by underscores
        safe_language = language.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('.', '')
        output_filename = f"{safe_language}_combined.hdf5"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            with h5py.File(output_path, 'w') as h5file:
                data_group = h5file.create_group('data')
                
                for demo_idx, demo_data in enumerate(all_demos):
                    demo_group = data_group.create_group(f'demo_{demo_idx}')
                    
                    # Copy all datasets except metadata
                    for key, value in demo_data.items():
                        if not key.startswith('_'):
                            if isinstance(value, np.ndarray):
                                demo_group.create_dataset(key, data=value)
                            else:
                                demo_group.create_dataset(key, data=np.array(value))
                    
                    # Add metadata
                    demo_group.attrs['source_file'] = demo_data.get('_source_file', 'unknown')
                    demo_group.attrs['source_demo_key'] = demo_data.get('_source_demo_key', 'unknown')
                    demo_group.attrs['combined_demo_idx'] = demo_idx
            
            print(f"✅ Saved combined file: {output_path} ({len(all_demos)} demos)")
            
            # Store comprehensive statistics
            combined_stats[language] = {
                'input_files': len(files),
                'total_demos': total_demos,
                'combined_demos': len(all_demos),
                'output_file': output_filename,
                'step_statistics': step_stats,
                'source_files': [os.path.basename(f) for f in files]
            }
            
        except Exception as e:
            print(f"❌ Error saving combined file for '{language}': {e}")
    
    return combined_stats


def save_analysis_results(stats: Dict[str, Any], output_dir: str):
    """
    Save analysis results and statistics.
    
    Args:
        stats: Statistics dictionary from combine_demos_by_language
        output_dir: Output directory
    """
    # Save detailed statistics as JSON
    stats_file = os.path.join(output_dir, "combination_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved combination statistics: {stats_file}")
    
    # Create concise stats file for easy reading
    concise_stats = []
    for language, data in stats.items():
        step_stats = data.get('step_statistics', {})
        concise_stats.append({
            'skill_language': language,
            'num_demos': data['combined_demos'],
            'min_steps': step_stats.get('min_steps', 'N/A'),
            'max_steps': step_stats.get('max_steps', 'N/A'),
            'mean_steps': step_stats.get('mean_steps', 'N/A'),
            'total_steps': step_stats.get('total_steps', 'N/A'),
            'source_files_count': data['input_files'],
            'output_file': data['output_file']
        })
    
    # Sort by number of demos (descending)
    concise_stats.sort(key=lambda x: x['num_demos'], reverse=True)
    
    concise_stats_file = os.path.join(output_dir, "concise_stats.json")
    with open(concise_stats_file, 'w') as f:
        json.dump(concise_stats, f, indent=2)
    print(f"Saved concise statistics: {concise_stats_file}")
    
    # Create summary
    total_input_files = sum(s['input_files'] for s in stats.values())
    total_combined_demos = sum(s['combined_demos'] for s in stats.values())
    
    summary = {
        'total_input_files': total_input_files,
        'total_languages': len(stats),
        'total_combined_demos': total_combined_demos,
        'compression_ratio': total_input_files / len(stats) if len(stats) > 0 else 0,
        'languages': list(stats.keys()),
        'concise_stats': concise_stats
    }
    
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_file}")
    
    # Print summary
    print(f"\n=== Combination Summary ===")
    print(f"Input files: {total_input_files}")
    print(f"Unique languages: {len(stats)}")
    print(f"Combined demos: {total_combined_demos}")
    print(f"Compression ratio: {summary['compression_ratio']:.2f}x")
    print(f"Output directory: {output_dir}")
    
    # Print concise stats table
    print(f"\n=== Concise Statistics ===")
    print(f"{'Skill Language':<40} {'Demos':<6} {'Min':<4} {'Max':<4} {'Mean':<6} {'Total':<6}")
    print("-" * 80)
    for stat in concise_stats:
        print(f"{stat['skill_language'][:39]:<40} {stat['num_demos']:<6} {stat['min_steps']:<4} {stat['max_steps']:<4} {stat['mean_steps']:<6} {stat['total_steps']:<6}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine atomic skills HDF5 files by language description (simplified version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine skills by parsing BDDL files directly
  python combine_skills_by_language_simple.py --input_dir datasets/hdf5_datasets/atomic_local_demos/closer_pick --output_dir datasets/hdf5_datasets/atomic_local_demos/closer_pick_combined
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing input HDF5 files'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save combined HDF5 files'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    print("=== Atomic Skills Language Combination (Simplified) ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Combine demos by language
    stats = combine_demos_by_language(args.input_dir, args.output_dir)
    
    if stats:
        # Save analysis results
        save_analysis_results(stats, args.output_dir)
        print(f"\n✅ Successfully combined skills by language!")
    else:
        print(f"\n❌ No skills were combined.")


if __name__ == "__main__":
    main()
