#!/usr/bin/env python3
"""
Combine Atomic Skills by Language

This script combines atomic skills HDF5 files based on their BDDL language descriptions.
It groups files by identical language and combines all demos into single HDF5 files.
"""

import os
import argparse
import json
import re
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import pickle


def extract_language_from_bddl(bddl_file_path: str) -> str:
    """Extract the language description from a BDDL file."""
    try:
        with open(bddl_file_path, 'r') as f:
            content = f.read()
        
        # Look for (:language ...) pattern
        match = re.search(r'\(:language\s+(.*?)\)', content)
        if match:
            language = match.group(1).strip()
            return language
        else:
            print(f"Warning: No language found in {bddl_file_path}")
            return None
    except FileNotFoundError:
        print(f"Warning: BDDL file not found: {bddl_file_path}")
        return None
    except Exception as e:
        print(f"Error reading BDDL file {bddl_file_path}: {e}")
        return None


def safe_filename(text: str) -> str:
    """Convert text to safe filename by replacing spaces and special characters."""
    # Replace spaces with underscores and remove special characters
    safe_name = re.sub(r'[^\w\s-]', '', text.strip())
    safe_name = re.sub(r'\s+', '_', safe_name)
    return safe_name.lower()


def get_hdf5_to_language_mapping(input_dir: str, bddl_dir: str) -> Dict[str, str]:
    """Create mapping from HDF5 files to their language descriptions."""
    mapping = {}
    
    # Get all HDF5 files in the input directory
    hdf5_files = list(Path(input_dir).glob("*_demo.hdf5"))
    
    for hdf5_file in hdf5_files:
        # Extract skill name from filename (remove _demo.hdf5 suffix)
        skill_name = hdf5_file.stem.replace("_demo", "")
        
        # Construct BDDL file path
        bddl_file_path = os.path.join(bddl_dir, f"{skill_name}.bddl")
        
        # Extract language from BDDL file
        language = extract_language_from_bddl(bddl_file_path)
        
        if language:
            mapping[str(hdf5_file)] = language
            print(f"Mapped {hdf5_file.name} -> '{language}'")
        else:
            print(f"Warning: Could not extract language for {hdf5_file.name}")
    
    return mapping


def analyze_hdf5_demos(hdf5_file_path: str) -> Dict[str, Any]:
    """Analyze an HDF5 file to extract demo statistics."""
    stats = {
        'file_path': hdf5_file_path,
        'num_demos': 0,
        'demo_steps': [],
        'total_steps': 0
    }
    
    try:
        with h5py.File(hdf5_file_path, 'r') as f:
            data_group = f['data']
            demo_keys = [key for key in data_group.keys() if key.startswith('demo_')]
            stats['num_demos'] = len(demo_keys)
            
            for demo_key in demo_keys:
                demo = data_group[demo_key]
                if 'actions' in demo:
                    steps = len(demo['actions'])
                    stats['demo_steps'].append(steps)
                    stats['total_steps'] += steps
    except Exception as e:
        print(f"Error analyzing {hdf5_file_path}: {e}")
        
    return stats


def combine_hdf5_files(hdf5_files: List[str], output_file: str) -> Dict[str, Any]:
    """Combine multiple HDF5 files into one."""
    combined_stats = {
        'source_files': hdf5_files,
        'num_source_files': len(hdf5_files),
        'total_demos': 0,
        'demo_steps': [],
        'total_steps': 0,
        'min_steps': float('inf'),
        'max_steps': 0,
        'mean_steps': 0.0
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    demo_counter = 0
    
    with h5py.File(output_file, 'w') as out_f:
        # Create data group
        out_data_group = out_f.create_group('data')
        
        # Process each input file
        for file_path in hdf5_files:
            try:
                with h5py.File(file_path, 'r') as in_f:
                    in_data_group = in_f['data']
                    demo_keys = [key for key in in_data_group.keys() if key.startswith('demo_')]
                    
                    print(f"  Processing {os.path.basename(file_path)} ({len(demo_keys)} demos)")
                    
                    for demo_key in demo_keys:
                        demo = in_data_group[demo_key]
                        
                        # Create new demo in output file
                        new_demo_key = f'demo_{demo_counter}'
                        new_demo = out_data_group.create_group(new_demo_key)
                        
                        # Copy all datasets from input demo to output demo
                        for dataset_name in demo.keys():
                            if isinstance(demo[dataset_name], h5py.Group):
                                # Handle obs group
                                new_subgroup = new_demo.create_group(dataset_name)
                                for sub_key in demo[dataset_name].keys():
                                    new_subgroup.create_dataset(
                                        sub_key, 
                                        data=demo[dataset_name][sub_key][:]
                                    )
                            else:
                                # Handle regular datasets
                                new_demo.create_dataset(
                                    dataset_name, 
                                    data=demo[dataset_name][:]
                                )
                        
                        # Update statistics
                        if 'actions' in demo:
                            steps = len(demo['actions'])
                            combined_stats['demo_steps'].append(steps)
                            combined_stats['total_steps'] += steps
                            combined_stats['min_steps'] = min(combined_stats['min_steps'], steps)
                            combined_stats['max_steps'] = max(combined_stats['max_steps'], steps)
                        
                        demo_counter += 1
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        combined_stats['total_demos'] = demo_counter
        if combined_stats['demo_steps']:
            combined_stats['mean_steps'] = np.mean(combined_stats['demo_steps'])
            if combined_stats['min_steps'] == float('inf'):
                combined_stats['min_steps'] = 0
        
        # Store metadata as attributes
        out_f.attrs['source_files'] = [os.path.basename(f) for f in hdf5_files]
        out_f.attrs['num_source_files'] = len(hdf5_files)
        out_f.attrs['total_demos'] = demo_counter
        out_f.attrs['total_steps'] = combined_stats['total_steps']
        
    return combined_stats


def main():
    parser = argparse.ArgumentParser(description='Combine atomic skills HDF5 files by language')
    parser.add_argument('--input_dir', required=True, 
                      help='Input directory containing HDF5 files')
    parser.add_argument('--output_dir', required=True,
                      help='Output directory for combined HDF5 files')
    parser.add_argument('--bddl_dir', default='externals/boss/libero/libero/bddl_files/atomic_skills',
                      help='Directory containing BDDL files')
    
    args = parser.parse_args()
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"BDDL directory: {args.bddl_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Create mapping from HDF5 files to languages
    print("\n=== Step 1: Creating HDF5 to Language Mapping ===")
    hdf5_to_language = get_hdf5_to_language_mapping(args.input_dir, args.bddl_dir)
    
    if not hdf5_to_language:
        print("Error: No valid HDF5-to-language mappings found!")
        return
    
    print(f"Found {len(hdf5_to_language)} valid mappings")
    
    # Step 2: Group files by language
    print("\n=== Step 2: Grouping Files by Language ===")
    language_to_files = defaultdict(list)
    
    for hdf5_file, language in hdf5_to_language.items():
        language_to_files[language].append(hdf5_file)
    
    print(f"Found {len(language_to_files)} unique languages:")
    for language, files in language_to_files.items():
        print(f"  '{language}': {len(files)} files")
    
    # Step 3: Combine files by language
    print("\n=== Step 3: Combining Files by Language ===")
    
    all_stats = {}
    combination_stats = {}
    
    for language, files in language_to_files.items():
        safe_lang_name = safe_filename(language)
        output_filename = f"{safe_lang_name}_combined.hdf5"
        output_path = os.path.join(args.output_dir, output_filename)
        
        print(f"\nCombining {len(files)} files for language: '{language}'")
        print(f"Output file: {output_filename}")
        
        # Combine the files
        stats = combine_hdf5_files(files, output_path)
        all_stats[language] = stats
        combination_stats[safe_lang_name] = {
            'language': language,
            'output_file': output_filename,
            'num_source_files': stats['num_source_files'],
            'total_demos': stats['total_demos'],
            'total_steps': stats['total_steps'],
            'min_steps': stats['min_steps'],
            'max_steps': stats['max_steps'],
            'mean_steps': stats['mean_steps'],
            'source_files': [os.path.basename(f) for f in stats['source_files']]
        }
        
        print(f"  Combined {stats['total_demos']} demos from {stats['num_source_files']} files")
        print(f"  Total steps: {stats['total_steps']}")
        print(f"  Step range: {stats['min_steps']}-{stats['max_steps']} (mean: {stats['mean_steps']:.1f})")
    
    # Step 4: Generate statistics files
    print("\n=== Step 4: Generating Statistics ===")
    
    # Detailed stats
    detailed_stats_file = os.path.join(args.output_dir, 'combination_stats.json')
    with open(detailed_stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"Detailed statistics saved to: {detailed_stats_file}")
    
    # Concise stats
    concise_stats_file = os.path.join(args.output_dir, 'concise_stats.json')
    with open(concise_stats_file, 'w') as f:
        json.dump(combination_stats, f, indent=2, default=str)
    print(f"Concise statistics saved to: {concise_stats_file}")
    
    # Summary stats
    total_input_files = len(hdf5_to_language)
    total_output_files = len(language_to_files)
    compression_ratio = total_input_files / total_output_files if total_output_files > 0 else 0
    total_demos = sum(stats['total_demos'] for stats in all_stats.values())
    total_steps = sum(stats['total_steps'] for stats in all_stats.values())
    
    summary = {
        'input_files': total_input_files,
        'output_files': total_output_files,
        'compression_ratio': compression_ratio,
        'total_demos': total_demos,
        'total_steps': total_steps,
        'unique_languages': list(language_to_files.keys())
    }
    
    summary_file = os.path.join(args.output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary statistics saved to: {summary_file}")
    
    # Print final summary
    print("\n=== Final Summary ===")
    print(f"Input files: {total_input_files}")
    print(f"Output files: {total_output_files}")
    print(f"Compression ratio: {compression_ratio:.1f}x")
    print(f"Total demos: {total_demos}")
    print(f"Total steps: {total_steps}")
    print(f"Unique languages: {len(language_to_files)}")
    
    print(f"\nCombined files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()