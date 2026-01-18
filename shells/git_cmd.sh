#!/bin/zsh

# Git commands for pushing VLM pose prediction reorganization
# This script adds the new vlm-pose/ directory and other important updates

echo "=== Setting up .gitignore for large folders ==="

# Add large folders to .gitignore if not already present
if ! grep -q "runs/" .gitignore 2>/dev/null; then
    echo "" >> .gitignore
    echo "# Large data folders" >> .gitignore
    echo "runs/" >> .gitignore
    echo "datasets/" >> .gitignore
    echo "rollouts/" >> .gitignore
    echo "wandb/" >> .gitignore
    echo "logs/" >> .gitignore
    echo "*.pt" >> .gitignore
    echo "debug_inits_imgs/" >> .gitignore
    echo "imgs/" >> .gitignore
    echo "outs/" >> .gitignore
    echo "Added large folders to .gitignore"
else
    echo ".gitignore already contains large folder exclusions"
fi

echo ""
echo "=== Adding files to git ==="

# Add the new vlm-pose directory and other important files
git add vlm-pose/
git add .gitignore
git add vla-evaluation/
git add vla-scripts/
git add utils/
git add shells/
git add scripts/
git add experiments/
git add *.md
git add *.py
git add *.toml
git add *.txt

echo "Added vlm-pose/ and other important files"

echo ""
echo "=== Checking git status ==="
git status --porcelain

echo ""
echo "=== Committing changes ==="

# Commit the changes
git commit -m "Reorganize VLM pose prediction components into vlm-pose/ directory

- Create clean vlm-pose/ structure with training/, evaluation/, data_preparation/, visualization/, utils/
- Copy VLM scripts from scattered locations to organized structure
- Update README files with zsh commands
- Keep original files intact for backward compatibility
- Add comprehensive documentation for each component"

echo "Committed changes"

echo ""
echo "=== Checking current branch ==="
git branch --show-current

echo ""
echo "=== Available branches ==="
git branch -a

echo ""
echo "=== Ready to push! ==="
echo "To push to current branch: git push"
echo "To push to specific branch: git push origin <branch-name>"
echo "To push all branches: git push --all origin"

echo ""
echo "=== Pushing all branches ==="
echo "Pushing main branch..."
git push origin main

echo "Pushing pose-generator branch..."
git push origin pose-generator

echo "Pushing reorganize-vlm-pose-predictor branch..."
git push origin reorganize-vlm-pose-predictor

echo ""
echo "=== All branches pushed successfully! ==="
echo "You can also use: git push --all origin (to push all branches at once)" 