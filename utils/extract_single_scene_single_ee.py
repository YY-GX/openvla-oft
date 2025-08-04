import os
import csv
import argparse
import json
from collections import defaultdict, Counter

INPUT_SPLIT_DIR = "datasets/local_pairs_datasets/splits"
OUTPUT_SPLIT_DIR = "datasets/local_pairs_datasets/smaller_splits"


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
    scene_counter = Counter()
    demo_to_scene = {}

    # First pass: find the most common scene and collect all rows
    for split_file in split_files:
        rows, _ = load_csv(os.path.join(args.input_dir, split_file))
        for row in rows:
            scene = parse_scene_name(row)
            if scene:
                demo_id = (row["source_skill_file_name"], row["source_demo_idx"])
                demo_to_scene[demo_id] = scene
                scene_counter[scene] += 1
        all_rows.extend(rows)
    most_common_scene, _ = scene_counter.most_common(1)[0]
    print(f"Most common scene: {most_common_scene}")

    # Second pass: filter to most common scene and deduplicate
    filtered_rows = [r for r in all_rows if parse_scene_name(r) == most_common_scene]
    
    # Group by (demo, contact, overview_image_idx)
    group_dict = defaultdict(list)
    for r in filtered_rows:
        key = (r["source_skill_file_name"], r["source_demo_idx"], r["pre_contact_number"], r["overview_image_idx"])
        group_dict[key].append(r)
    
    # For each group, keep the row with the pose_timestep closest to 5 steps before contact
    # The poses are extracted from [contact_timestep - 8, contact_timestep - 7, contact_timestep - 6, contact_timestep - 5, contact_timestep - 4, contact_timestep - 3]
    # We want the one at contact_timestep - 5, which is the 4th pose (index 3) in the range
    deduped_rows = []
    for group in group_dict.values():
        # Sort by pose_timestep and take the 4th one (index 3) which should be 5 steps before contact
        sorted_group = sorted(group, key=lambda r: int(r["pose_timestep"]))
        if len(sorted_group) >= 4:
            target_row = sorted_group[3]  # 4th pose (index 3)
        else:
            # Fallback: if we don't have 4 poses, take the middle one
            target_row = sorted_group[len(sorted_group) // 2]
        deduped_rows.append(target_row)
    
    print(f"Total deduplicated rows: {len(deduped_rows)}")
    
    # Get unique demos for splitting
    unique_demos = list(set((r["source_skill_file_name"], r["source_demo_idx"]) for r in deduped_rows))
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
    train_rows = [r for r in deduped_rows if (r["source_skill_file_name"], r["source_demo_idx"]) in train_demos]
    val_rows = [r for r in deduped_rows if (r["source_skill_file_name"], r["source_demo_idx"]) in val_demos]
    test_rows = [r for r in deduped_rows if (r["source_skill_file_name"], r["source_demo_idx"]) in test_demos]
    
    # Write splits
    fieldnames = deduped_rows[0].keys() if deduped_rows else []
    stats = {}
    
    for split_name, split_rows in [("train_annotation.csv", train_rows), 
                                   ("val_annotation.csv", val_rows), 
                                   ("test_annotation.csv", test_rows)]:
        out_path = os.path.join(args.output_dir, split_name)
        write_csv(out_path, split_rows, fieldnames)
        
        # Stats
        demo_keys = set((r["source_skill_file_name"], r["source_demo_idx"]) for r in split_rows)
        stats[split_name] = {
            "num_rows": len(split_rows),
            "num_demos": len(demo_keys),
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