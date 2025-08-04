import os
import csv
import argparse
import json
from collections import defaultdict, Counter

INPUT_SPLIT_DIR = "datasets/local_pairs_datasets/splits"
OUTPUT_SPLIT_DIR = "datasets/local_pairs_datasets/hungarian_splits"

# The 6 skills from KITCHEN_SCENE4 that are used in smaller_splits
KITCHEN_SCENE4_SKILLS = [
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet", 
    "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet",
    "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer"
]

def parse_scene_name(row):
    import re
    # Try to extract scene from language_description or source_skill_file_name
    for key in ["language_description", "source_skill_file_name", "task_name"]:
        if key in row:
            # Look for pattern like KITCHEN_SCENE10, LIVING_ROOM_SCENE1, etc.
            match = re.search(r'([A-Z_]+_SCENE\d+)', row[key])
            if match:
                return match.group(1)
    # fallback: try to find any SCENE pattern in any field
    for v in row.values():
        match = re.search(r'([A-Z_]+_SCENE\d+)', str(v))
        if match:
            return match.group(1)
    return None

def load_csv(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, reader.fieldnames

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=INPUT_SPLIT_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_SPLIT_DIR)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    split_files = ["train_annotation.csv", "val_annotation.csv", "test_annotation.csv"]
    all_rows = []

    # Load all rows from splits
    for split_file in split_files:
        rows, _ = load_csv(os.path.join(args.input_dir, split_file))
        all_rows.extend(rows)
    
    print(f"Total rows loaded: {len(all_rows)}")
    
    # Filter to KITCHEN_SCENE4 and the 6 specific skills
    filtered_rows = []
    for row in all_rows:
        scene = parse_scene_name(row)
        skill = row.get("language_description", "")
        
        if scene == "KITCHEN_SCENE4" and skill in KITCHEN_SCENE4_SKILLS:
            filtered_rows.append(row)
    
    print(f"Filtered to KITCHEN_SCENE4 with 6 skills: {len(filtered_rows)} rows")
    
    # Group by demo to ensure we have all 6 poses per demo
    demo_to_rows = defaultdict(list)
    for row in filtered_rows:
        demo_id = (row["source_skill_file_name"], row["source_demo_idx"])
        demo_to_rows[demo_id].append(row)
    
    # Keep only demos that have all 6 poses (ee_pose_idx should be 6 consecutive indices)
    complete_demos = {}
    for demo_id, rows in demo_to_rows.items():
        # Group by (pre_contact_number, overview_image_idx) to get the 6 poses
        pose_groups = defaultdict(list)
        for row in rows:
            key = (row["pre_contact_number"], row["overview_image_idx"])
            pose_groups[key].append(row)
        
        # Find the group with 6 poses (should be the main contact group)
        for group_key, group_rows in pose_groups.items():
            if len(group_rows) >= 6:
                # Sort by ee_pose_idx and take the first 6
                sorted_rows = sorted(group_rows, key=lambda r: int(r["ee_pose_idx"]))
                if len(sorted_rows) >= 6:
                    complete_demos[demo_id] = sorted_rows[:6]  # Keep first 6 poses
                    break  # Use the first group that has 6 poses
    
    print(f"Found {len(complete_demos)} demos with all 6 poses")
    
    # Flatten back to rows
    all_complete_rows = []
    for demo_rows in complete_demos.values():
        all_complete_rows.extend(demo_rows)
    
    print(f"Total rows with complete demos: {len(all_complete_rows)}")
    
    # Get unique demos for splitting
    unique_demos = list(complete_demos.keys())
    unique_demos.sort()  # Sort for reproducible splits
    print(f"Total unique demos: {len(unique_demos)}")
    
    # Split demos according to ratios
    import random
    random.seed(42)  # For reproducible splits
    random.shuffle(unique_demos)
    
    train_end = int(len(unique_demos) * args.train_ratio)
    val_end = train_end + int(len(unique_demos) * args.val_ratio)
    
    train_demos = set(unique_demos[:train_end])
    val_demos = set(unique_demos[train_end:val_end])
    test_demos = set(unique_demos[val_end:])
    
    print(f"Train demos: {len(train_demos)}")
    print(f"Val demos: {len(val_demos)}")
    print(f"Test demos: {len(test_demos)}")
    
    # Split rows based on demo assignment
    train_rows = [r for r in all_complete_rows if (r["source_skill_file_name"], r["source_demo_idx"]) in train_demos]
    val_rows = [r for r in all_complete_rows if (r["source_skill_file_name"], r["source_demo_idx"]) in val_demos]
    test_rows = [r for r in all_complete_rows if (r["source_skill_file_name"], r["source_demo_idx"]) in test_demos]
    
    # Write splits
    fieldnames = all_complete_rows[0].keys() if all_complete_rows else []
    stats = {}
    
    for split_name, split_rows in [("train_annotation.csv", train_rows), 
                                   ("val_annotation.csv", val_rows), 
                                   ("test_annotation.csv", test_rows)]:
        out_path = os.path.join(args.output_dir, split_name)
        write_csv(out_path, split_rows, fieldnames)
        
        # Stats
        demo_keys = set((r["source_skill_file_name"], r["source_demo_idx"]) for r in split_rows)
        skill_keys = set(r["language_description"] for r in split_rows)
        stats[split_name] = {
            "num_rows": len(split_rows),
            "num_demos": len(demo_keys),
            "num_skills": len(skill_keys),
            "avg_rows_per_demo": len(split_rows) / max(1, len(demo_keys)),
        }
        print(f"{split_name}: {stats[split_name]}")

    # Copy split_config.json
    with open(os.path.join(args.input_dir, "split_config.json"), "r") as f:
        split_cfg = json.load(f)
    with open(os.path.join(args.output_dir, "split_config.json"), "w") as f:
        json.dump(split_cfg, f, indent=2)
    
    # Save stats
    with open(os.path.join(args.output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {os.path.join(args.output_dir, 'stats.json')}")

if __name__ == "__main__":
    main() 