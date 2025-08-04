import os
import csv
import argparse
import json
import random
from collections import defaultdict

"""
  KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet: 47 demos
  KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet: 47 demos
  KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet: 48 demos
  KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet: 45 demos
  KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack: 38 demos
  KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer: 33 demos
Skill 'KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet': 32 train, 7 val, 8 test demos
Skill 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet': 32 train, 7 val, 8 test demos
Skill 'KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet': 33 train, 7 val, 8 test demos
Skill 'KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet': 31 train, 6 val, 8 test demos
Skill 'KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack': 26 train, 5 val, 7 test demos
Skill 'KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer': 23 train, 4 val, 6 test demos
train_annotation.csv: {'num_rows': 17959, 'num_demos': 177, 'num_skills': 6, 'avg_rows_per_demo': 101.46327683615819}
val_annotation.csv: {'num_rows': 3777, 'num_demos': 36, 'num_skills': 6, 'avg_rows_per_demo': 104.91666666666667}
test_annotation.csv: {'num_rows': 5774, 'num_demos': 45, 'num_skills': 6, 'avg_rows_per_demo': 128.3111111111111}
Stats saved to datasets/local_pairs_datasets/debug_splits/stats.json
"""


INPUT_SPLIT_DIR = "datasets/local_pairs_datasets/smaller_splits"
OUTPUT_SPLIT_DIR = "datasets/local_pairs_datasets/debug_splits"


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

    # Load all annotations from smaller_splits
    split_files = ["train_annotation.csv", "val_annotation.csv", "test_annotation.csv"]
    all_rows = []
    
    for split_file in split_files:
        rows, _ = load_csv(os.path.join(args.input_dir, split_file))
        all_rows.extend(rows)
    
    print(f"Total rows loaded: {len(all_rows)}")
    
    # Group by skill (language_description)
    skill_to_demos = defaultdict(list)
    for row in all_rows:
        skill = row["language_description"]
        demo_id = (row["source_skill_file_name"], row["source_demo_idx"])
        skill_to_demos[skill].append((demo_id, row))
    
    print(f"Found {len(skill_to_demos)} unique skills:")
    for skill, demos in skill_to_demos.items():
        print(f"  {skill}: {len(set(demo_id for demo_id, _ in demos))} demos")
    
    # Split each skill's demos
    train_rows = []
    val_rows = []
    test_rows = []
    
    random.seed(42)  # For reproducible splits
    
    for skill, demo_data in skill_to_demos.items():
        # Get unique demos for this skill
        unique_demos = list(set(demo_id for demo_id, _ in demo_data))
        random.shuffle(unique_demos)
        
        # Split demos
        train_end = int(len(unique_demos) * args.train_ratio)
        val_end = train_end + int(len(unique_demos) * args.val_ratio)
        
        train_demos = set(unique_demos[:train_end])
        val_demos = set(unique_demos[train_end:val_end])
        test_demos = set(unique_demos[val_end:])
        
        print(f"Skill '{skill}': {len(train_demos)} train, {len(val_demos)} val, {len(test_demos)} test demos")
        
        # Assign rows to splits based on demo assignment
        for demo_id, row in demo_data:
            if demo_id in train_demos:
                train_rows.append(row)
            elif demo_id in val_demos:
                val_rows.append(row)
            elif demo_id in test_demos:
                test_rows.append(row)
    
    # Write splits
    fieldnames = all_rows[0].keys() if all_rows else []
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