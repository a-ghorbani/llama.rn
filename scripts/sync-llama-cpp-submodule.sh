#!/bin/bash
set -euo pipefail

STAGING_BRANCH="auto/sync-llama.cpp-staging"
LLAMA_DIR="llama.cpp"

echo "🌱 Preparing staging branch: $STAGING_BRANCH"
git fetch origin main

# Clean up any existing staging branch to ensure fresh start
git push origin --delete "$STAGING_BRANCH" 2>/dev/null || echo "No existing staging branch to delete"
git branch -D "$STAGING_BRANCH" 2>/dev/null || echo "No local staging branch to delete"

# Check if persistent sync branch exists and use it as base, otherwise use main
WORK_BRANCH="auto/sync-llama.cpp"
if git show-ref --verify --quiet refs/remotes/origin/"$WORK_BRANCH"; then
  echo "📦 Using existing sync branch as base: $WORK_BRANCH"
  git fetch origin "$WORK_BRANCH"
  git checkout -B "$STAGING_BRANCH" origin/"$WORK_BRANCH"
else
  echo "🆕 Using main as base (no existing sync branch)"
  git checkout -B "$STAGING_BRANCH" origin/main
fi

echo "🔍 Checking latest llama.cpp release..."
LATEST_TAG=$(curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | jq -r .tag_name)

if [[ -z "$LATEST_TAG" || "$LATEST_TAG" == "null" ]]; then
  echo "❌ Failed to fetch latest tag"
  exit 1
fi

cd "$LLAMA_DIR"
CURRENT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "none")
cd ..

echo "📌 Latest tag: $LATEST_TAG"
echo "📦 Current tag in llama.cpp: $CURRENT_TAG"

if [[ "$LATEST_TAG" == "$CURRENT_TAG" ]]; then
  echo "✅ Already synced to $LATEST_TAG"
  echo "🛠 Running bootstrap to ensure cpp/ directory is up to date..."
  yarn bootstrap

  # Check if bootstrap created any changes
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "💾 Committing bootstrap changes..."
    git add -A
    git commit -m "chore(sync): update cpp/ directory with bootstrap (no llama.cpp version change)"
  fi

  # Still need to push the staging branch for the workflow to continue
  git push origin "$STAGING_BRANCH"
  exit 0
fi

echo "📥 Updating llama.cpp to $LATEST_TAG..."
cd "$LLAMA_DIR"
git fetch --tags
git checkout "refs/tags/$LATEST_TAG"
cd ..

git add "$LLAMA_DIR"
git commit -m "chore: update llama.cpp to $LATEST_TAG (submodule ref)"

echo "🛠 Running bootstrap to copy files and apply patches..."
yarn bootstrap

# Check if bootstrap created any changes in cpp/ directory
if git diff --quiet && git diff --cached --quiet; then
  echo "✅ No changes after bootstrap — cpp/ directory already up to date."
else
  echo "💾 Committing bootstrap changes..."
  git add -A
  git commit -m "chore(sync): update cpp/ directory after llama.cpp $LATEST_TAG bootstrap"
fi

git push origin "$STAGING_BRANCH"

echo "🚀 Submodule updated, bootstrap completed, and committed to staging branch"
