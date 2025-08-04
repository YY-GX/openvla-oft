import sys
import os
import random
import json

if len(sys.argv) < 6:
    print("Usage: python generate_sample_indices.py <sample_size> <seed> <train_total> <val_total> <test_total>")
    sys.exit(1)

sample_size = int(sys.argv[1])
seed = int(sys.argv[2])
train_total = int(sys.argv[3])
val_total = int(sys.argv[4])
test_total = int(sys.argv[5])

splits = [
    ("train", train_total),
    ("val", val_total),
    ("test", test_total),
]

random.seed(seed)
indices_dict = {}
for split, total in splits:
    indices = random.sample(range(total), sample_size)
    indices_dict[split] = indices

os.makedirs("runs/eval_results/vlm_pose_generator", exist_ok=True)
output_path = "runs/eval_results/vlm_pose_generator/sample_indices.json"
with open(output_path, 'w') as f:
    json.dump(indices_dict, f)
print(f"Saved sample indices for all splits to {output_path}") 