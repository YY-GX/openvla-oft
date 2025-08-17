#!/usr/bin/env python3
"""
Cleanup script to remove test files that are no longer needed.
"""

import os
import shutil

def cleanup_test_files():
    """Remove test files that should be deleted."""
    
    # Files to delete
    files_to_delete = [
        "single_skill_tasks_test.json",
        "single_skill_tasks_minimal.json", 
        "contact_timesteps_test.json",
        "test_quick_fixes.py",
        "cleanup_test_files.py"
    ]
    
    # Directories to delete
    dirs_to_delete = [
        "pose_similarity_results"
    ]
    
    print("=== Cleaning up test files ===")
    
    # Delete files
    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)
            print(f"✓ Deleted: {file}")
        else:
            print(f"- Not found: {file}")
    
    # Delete directories
    for dir in dirs_to_delete:
        if os.path.exists(dir):
            shutil.rmtree(dir)
            print(f"✓ Deleted directory: {dir}")
        else:
            print(f"- Not found: {dir}")
    
    print("\n=== Cleanup complete ===")

if __name__ == "__main__":
    cleanup_test_files() 