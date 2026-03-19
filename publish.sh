#!/bin/bash

# Configuration
REPO_BRANCH="main"
OUTPUT_DIR="output"
STATS_FILE="$OUTPUT_DIR/stats_results.txt"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "🚗 Generating Muni Metro Fatal Crashes Map..."
python3 analyze_and_map.py

echo "🚑 Generating Muni Metro Injury Crashes Map..."
python3 analyze_injuries.py

echo "📊 Running Statistical Analysis and saving to $STATS_FILE..."
python3 stats_analysis.py > "$STATS_FILE"
echo "Done! Stats summary available at $STATS_FILE"

echo "🌐 Publishing changes to GitHub..."

# Add generation outputs
git add Muni_Metro_Fatal_Crashes.html
git add Muni_Metro_Injury_Crashes.html
git add "$STATS_FILE"
git add README.md
git add requirements.txt

# Commit and Push
git commit -m "chore: auto-update maps and statistics via publish.sh"
git push origin "$REPO_BRANCH"

echo "✅ Successfully pushed to $REPO_BRANCH! If GitHub Pages is enabled, your maps are now living on the web."
