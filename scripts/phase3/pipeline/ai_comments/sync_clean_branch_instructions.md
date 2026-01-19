# Syncing atomic-skill-learning to atomic-skill-learning-clean

## Background
- `atomic-skill-learning` branch has 407k files (includes large .ply, outputs, etc.)
- `atomic-skill-learning-clean` branch has ~200 files (code only, no large files)
- Clean branch is for transferring code to other clusters via GitHub

## Folder Locations
- Main workspace: `/mnt/arc/yygx/pkgs_baselines/openvla-oft` (atomic-skill-learning)
- Clean worktree: `/mnt/arc/yygx/pkgs_baselines/openvla-oft-clean` (atomic-skill-learning-clean)

## Check What Changed
```bash
cd /mnt/arc/yygx/pkgs_baselines/openvla-oft-clean
git diff atomic-skill-learning --name-only
```

## Sync All Code Files at Once
```bash
cd /mnt/arc/yygx/pkgs_baselines/openvla-oft-clean
git checkout atomic-skill-learning -- \
  prismatic/ \
  vla-scripts/ \
  experiments/robot/ \
  scripts/phase3/pipeline/config/ \
  scripts/phase3/pipeline/evaluation/ \
  scripts/phase3/pipeline/data_generation/ \
  scripts/phase3/pipeline/motion_planning/ \
  scripts/phase3/pipeline/utils/ \
  shells/
git commit -m "Sync with atomic-skill-learning"
git push
```

## Clone on New Cluster
```bash
git clone https://github.com/YY-GX/openvla-oft.git
cd openvla-oft
git checkout atomic-skill-learning-clean
```
