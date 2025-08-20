#!/bin/bash
# Feature Hunter Initialization Script

set -e

BASE_DIR="$HOME/mtd-ai-security/features"
RUN_ID=$(date +"%Y%m%d_%H%M")
RUN_DIR="$BASE_DIR/runs/$RUN_ID"

echo "Setting up Feature Hunter in $RUN_DIR"

# Create directory structure
mkdir -p "$RUN_DIR"/{capture/{pcap,logs},features,exp_{clustering,autoencoder,contrastive,timeseries,rag/{kb},agents,feature_importance,generative},report/{figs},scripts,prep}

# Copy config to run directory
cp config.yaml "$RUN_DIR/"

# Copy scripts to run directory
cp scripts/*.sh "$RUN_DIR/scripts/"
chmod +x "$RUN_DIR/scripts/"*.sh

# Create symlink to latest run
rm -f "$BASE_DIR/latest"
ln -sf "runs/$RUN_ID" "$BASE_DIR/latest"

echo "Directory structure created:"
echo "  Base: $BASE_DIR"
echo "  Current run: $RUN_DIR"
echo "  Latest symlink: $BASE_DIR/latest"

# Check dependencies
echo "Checking Python dependencies..."
python3 -c "import sys; print(f'Python {sys.version}')"

# Install requirements
echo "Installing Python packages..."
pip3 install -r requirements.txt

echo "Setup complete! Current run: $RUN_ID"
echo "Export this for convenience: export FEATURE_HUNTER_RUN=$RUN_DIR"
