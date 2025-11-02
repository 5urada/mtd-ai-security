#!/bin/bash

# ID-HAM Setup Script for UV Package Manager
# Updated version using uv instead of pip

set -e  # Exit on error

echo "=================================="
echo "ID-HAM Setup with UV"
echo "=================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Source the shell configuration to get uv in PATH
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
fi

echo "✓ UV version: $(uv --version)"

# Create virtual environment with uv
echo ""
echo "Creating virtual environment..."
uv venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies with uv..."
uv pip install -r requirements.txt

echo ""
echo "=================================="
echo "✓ Setup Complete!"
echo "=================================="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run quick test:"
echo "  source .venv/bin/activate"
echo "  python run_experiments_quick.py"
echo ""
echo "To run full experiments:"
echo "  source .venv/bin/activate"
echo "  python run_experiments.py"
echo "=================================="