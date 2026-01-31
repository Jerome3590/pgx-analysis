# Git Workflow: Pushing Changes from EC2 Before Pulling

## Problem
When you have local changes on EC2 (e.g., `sed` commands to fix line endings or other modifications), running `git pull` can overwrite your uncommitted changes or cause conflicts.

## Solution: Always Commit and Push Before Pulling

### Step-by-Step Workflow

#### 1. Check What Has Changed
```bash
cd ~/pgx-analysis
git status
```

This shows:
- **Modified files**: Files you've changed locally
- **Untracked files**: New files not yet in Git
- **Staged files**: Files ready to commit

#### 2. Review Your Changes (Optional but Recommended)
```bash
# See what changed in specific files
git diff <filename>

# See all changes
git diff

# See untracked files
git status --untracked-files=all
```

#### 3. Stage Your Changes
```bash
# Stage all changes (modified + new files)
git add .

# OR stage specific files only
git add path/to/file1.sh path/to/file2.sh

# OR stage all changes in a directory
git add archived/utility_scripts/
```

#### 4. Commit Your Changes
```bash
# Commit with a descriptive message
git commit -m "EC2: Fix line endings with sed for shell scripts"

# OR if you have multiple types of changes, be more specific:
git commit -m "EC2: Fix line endings and update configuration"
```

**Good commit message examples:**
- `"EC2: Fix line endings in run_cohort_workflow.sh"`
- `"EC2: Update sed commands for Windows compatibility"`
- `"EC2: Local configuration changes"`

#### 5. Push to GitHub
```bash
git push origin main
```

**If push fails** (remote has new commits):
```bash
# First, pull with rebase to avoid merge commits
git pull --rebase origin main

# Resolve any conflicts if they occur, then:
git add .
git rebase --continue

# Finally, push
git push origin main
```

#### 6. Now Pull Latest Changes (if needed)
```bash
git pull origin main
```

## Common Scenarios

### Scenario 1: You Have Uncommitted Changes and Want to Pull

**Option A: Commit First (Recommended)**
```bash
git add .
git commit -m "EC2: Local changes before pull"
git pull origin main
# Resolve conflicts if any
git push origin main
```

**Option B: Stash Changes Temporarily**
```bash
# Save changes temporarily
git stash

# Pull latest
git pull origin main

# Restore your changes
git stash pop

# Now commit and push
git add .
git commit -m "EC2: Restored stashed changes"
git push origin main
```

### Scenario 2: Your `sed` Changes Keep Getting Overwritten

**Problem**: You run `sed -i 's/\r$//' run_*.sh` to fix line endings, but `git pull` overwrites them.

**Solution**: Commit the fixed files so they persist:
```bash
# Fix line endings
sed -i 's/\r$//' archived/utility_scripts/*.sh

# Commit the fixed files
git add archived/utility_scripts/*.sh
git commit -m "EC2: Fix line endings in shell scripts"

# Push to GitHub
git push origin main

# Now pull won't overwrite them (they're already committed)
git pull origin main
```

**Better Solution**: Add a `.gitattributes` file to handle line endings automatically:
```bash
# Create .gitattributes file
cat > .gitattributes << 'EOF'
# Auto detect text files and perform LF normalization
* text=auto

# Explicitly declare text files you want to always be normalized and converted
# to native line endings on checkout
*.sh text eol=lf
*.py text eol=lf
*.md text eol=lf

# Denote all files that are truly binary and should not be modified
*.parquet binary
*.csv binary
*.png binary
*.jpg binary
*.jpeg binary
*.gif binary
*.pdf binary
EOF

# Commit the .gitattributes file
git add .gitattributes
git commit -m "Add .gitattributes for consistent line endings"
git push origin main
```

### Scenario 3: You Want to See What Would Be Overwritten

```bash
# Check what files would be affected by pull
git fetch origin
git diff HEAD origin/main --name-only

# See actual differences
git diff HEAD origin/main
```

### Scenario 4: You Accidentally Pulled and Lost Changes

**If you haven't committed yet:**
```bash
# Check Git reflog for recent actions
git reflog

# Find the commit before the pull, then:
git reset --hard HEAD@{1}  # Replace {1} with the right number from reflog
```

**If changes are already lost:**
- Check if you have backups
- Re-apply your changes manually

## Best Practices

1. **Always commit before pulling** if you have local changes
2. **Use descriptive commit messages** so you know what changed
3. **Push frequently** to avoid losing work
4. **Use `.gitattributes`** to handle line endings automatically
5. **Check `git status`** before pulling to see if you have uncommitted changes

## Quick Reference Commands

```bash
# Check status
git status

# See changes
git diff

# Stage all changes
git add .

# Commit
git commit -m "Your message"

# Push
git push origin main

# Pull (after committing/pushing)
git pull origin main

# If conflicts occur during pull
git status          # See conflicted files
# Edit files to resolve conflicts
git add .
git commit -m "Resolve merge conflicts"
git push origin main
```

## Troubleshooting

### "Your branch is behind 'origin/main'"
```bash
git pull origin main
```

### "Your branch is ahead of 'origin/main'"
```bash
git push origin main
```

### "Updates were rejected because the remote contains work"
```bash
git pull --rebase origin main
# Resolve conflicts, then:
git push origin main
```

### "Please commit your changes or stash them"
```bash
# Option 1: Commit
git add .
git commit -m "Your changes"
git pull origin main

# Option 2: Stash
git stash
git pull origin main
git stash pop
```

