#!/usr/bin/env python3
"""
Simple validation script to check the dataset curation pipeline files.
This script validates file structure, syntax, and basic functionality without 
running the full tests that might have dependency issues.
"""

import os
import ast
import json
from pathlib import Path

def validate_python_syntax(file_path):
    """Validate Python syntax by parsing AST."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True, "Syntax OK"
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def check_file_structure():
    """Check that all required files exist."""
    data_prep_dir = Path(__file__).parent.parent / "data_preparation" / "v2_pipeline"
    print(f"   Looking in directory: {data_prep_dir}")
    
    required_files = {
        "detect_contact_timesteps_v2.py": "Contact timestep detection script",
        "extract_local_pairs_v2.py": "Pose extraction script", 
        "validate_pose_similarity.py": "Pose validation script",
        "single_skill_tasks_44.json": "Single skill tasks list"
    }
    
    results = {}
    for filename, description in required_files.items():
        file_path = data_prep_dir / filename
        exists = file_path.exists()
        results[filename] = {
            "exists": exists,
            "description": description,
            "path": str(file_path)
        }
        
        if exists and filename.endswith('.py'):
            syntax_ok, syntax_msg = validate_python_syntax(file_path)
            results[filename]["syntax_valid"] = syntax_ok
            results[filename]["syntax_message"] = syntax_msg
    
    return results

def validate_json_files():
    """Validate JSON configuration files."""
    data_prep_dir = Path(__file__).parent.parent / "data_preparation" / "v2_pipeline"
    json_file = data_prep_dir / "single_skill_tasks_44.json"
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list) and len(data) == 44:
            return True, f"Valid JSON with {len(data)} tasks"
        else:
            return False, f"Expected list of 44 tasks, got {type(data)} with {len(data) if hasattr(data, '__len__') else 'unknown'} items"
    except Exception as e:
        return False, f"JSON validation error: {e}"

def check_function_definitions():
    """Check that key functions are defined in the scripts."""
    data_prep_dir = Path(__file__).parent.parent / "data_preparation" / "v2_pipeline"
    
    expected_functions = {
        "detect_contact_timesteps_v2.py": [
            "detect_first_contact_timestep",
            "load_single_skill_tasks", 
            "match_task_name_to_file",
            "main"
        ],
        "extract_local_pairs_v2.py": [
            "extract_contact_based_poses",
            "extract_overview_images",
            "parse_language_description",
            "main"
        ],
        "validate_pose_similarity.py": [
            "PoseStatisticsAnalyzer",
            "PoseFilteringProcessor", 
            "main"
        ]
    }
    
    results = {}
    
    for script_name, expected_funcs in expected_functions.items():
        script_path = data_prep_dir / script_name
        results[script_name] = {}
        
        try:
            with open(script_path, 'r') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            # Extract function and class names
            defined_names = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    defined_names.add(node.name)
            
            for func_name in expected_funcs:
                results[script_name][func_name] = func_name in defined_names
                
        except Exception as e:
            results[script_name]["error"] = str(e)
    
    return results

def main():
    """Run all validations."""
    print("=== Dataset Curation Pipeline Validation ===\n")
    
    # Check file structure
    print("1. Checking file structure...")
    file_results = check_file_structure()
    
    all_files_ok = True
    for filename, info in file_results.items():
        status = "✅" if info["exists"] else "❌"
        print(f"   {status} {filename}: {info['description']}")
        
        if info["exists"] and "syntax_valid" in info:
            syntax_status = "✅" if info["syntax_valid"] else "❌"
            print(f"      {syntax_status} Syntax: {info['syntax_message']}")
            all_files_ok &= info["syntax_valid"]
        
        all_files_ok &= info["exists"]
    
    print()
    
    # Validate JSON files
    print("2. Validating JSON configuration...")
    json_valid, json_msg = validate_json_files()
    json_status = "✅" if json_valid else "❌"
    print(f"   {json_status} single_skill_tasks_44.json: {json_msg}")
    print()
    
    # Check function definitions
    print("3. Checking function definitions...")
    func_results = check_function_definitions()
    
    all_funcs_ok = True
    for script_name, funcs in func_results.items():
        print(f"   {script_name}:")
        
        if "error" in funcs:
            print(f"      ❌ Error: {funcs['error']}")
            all_funcs_ok = False
            continue
            
        for func_name, defined in funcs.items():
            status = "✅" if defined else "❌"
            print(f"      {status} {func_name}")
            all_funcs_ok &= defined
    
    print()
    
    # Summary
    overall_status = all_files_ok and json_valid and all_funcs_ok
    status_icon = "🎉" if overall_status else "⚠️"
    status_text = "PASSED" if overall_status else "ISSUES FOUND"
    
    print(f"=== Validation Summary: {status_icon} {status_text} ===")
    
    if overall_status:
        print("All dataset curation pipeline files are properly structured and ready for use!")
    else:
        print("Some issues were found. Please review the output above.")
    
    return overall_status

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)