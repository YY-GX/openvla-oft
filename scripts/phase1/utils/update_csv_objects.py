#!/usr/bin/env python3
import json
import csv
import os

def update_csv_objects():
    """Update the object column in skills_objects_mapping.csv using real object names from debug_target_detection.json"""
    
    # Read the debug target detection JSON file
    json_file = "datasets/debug_target_detection.json"
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found!")
        return
    
    with open(json_file, 'r') as f:
        target_detection = json.load(f)
    
    print(f"Loaded {len(target_detection)} object mappings from JSON")
    
    # Read the current CSV file
    csv_file = "skills_objects_mapping.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return
    
    # Read all rows from CSV
    rows = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            # Get the init files for this skill
            init_files_str = row['init_files']
            if init_files_str != 'No matching .init files found':
                # Split by semicolon to get individual init files
                init_files = [f.strip() for f in init_files_str.split(';')]
                
                # Find the most common object name among all init files for this skill
                object_counts = {}
                for init_file in init_files:
                    if init_file in target_detection:
                        obj_name = target_detection[init_file]
                        if obj_name:  # Skip null values
                            object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
                
                # Update the object column with the most common object name
                if object_counts:
                    # Get the most common object
                    most_common_obj = max(object_counts.items(), key=lambda x: x[1])[0]
                    row['object'] = most_common_obj
                    print(f"Updated '{row['skill_name']}' object from '{row['object']}' to '{most_common_obj}'")
                else:
                    print(f"Warning: No valid object names found for '{row['skill_name']}'")
            else:
                print(f"Skipping '{row['skill_name']}' - no init files found")
            
            rows.append(row)
    
    # Write the updated CSV file
    output_file = "skills_objects_mapping_updated.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nUpdated CSV saved to: {output_file}")
    
    # Also update the original file
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Original CSV file updated: {csv_file}")
    
    # Print summary of changes
    print(f"\nSummary of object updates:")
    for row in rows:
        if row['init_files'] != 'No matching .init files found':
            print(f"  {row['skill_name']}: {row['object']}")

if __name__ == "__main__":
    update_csv_objects()
