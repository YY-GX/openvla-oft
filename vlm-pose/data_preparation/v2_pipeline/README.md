# VLM Pose Dataset Curation Pipeline v2

This directory contains the improved dataset curation pipeline for creating high-quality pose VLM datasets.

## Key Improvements

- **Contact-Based Precision**: Extracts poses exactly [-4, -3, -2] steps before contact
- **Single Skill Focus**: Uses only 44 high-quality single skill tasks  
- **First Contact Detection**: Ignores subsequent contacts, focuses on first interaction
- **Pose Consistency Validation**: Ensures poses are similar within skills

## Pipeline Scripts

### 1. Contact Detection
```bash
python detect_contact_timesteps_v2.py
```
Detects first contact timesteps between robot and objects, saves to `contact_timesteps.json`.

### 2. Pose Extraction  
```bash
python extract_local_pairs_v2.py
```
Extracts poses at fixed window relative to contact timesteps. Creates pose-image pairs.

### 3. Pose Validation
```bash
# Statistics mode - analyze pose similarity
python validate_pose_similarity.py --mode statistics

# Filtering mode - remove inconsistent poses  
python validate_pose_similarity.py --mode filtering
```

## Data Files

- `single_skill_tasks_44.json` - List of 44 single skill task names
- `contact_timesteps.json` - Generated contact timesteps (after step 1)

## Output

- Pose datasets saved to `/datasets/local_pairs_datasets_v2/`
- Validation results saved to `pose_similarity_results/`

## Usage Pipeline

1. Run contact detection to generate `contact_timesteps.json`
2. Run pose extraction to create dataset
3. Run validation to analyze/filter poses (optional)

All scripts support `--help` for detailed parameter information.