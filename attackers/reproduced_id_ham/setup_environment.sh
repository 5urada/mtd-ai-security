#!/bin/bash

# ID-HAM Environment Setup Script
# Based on "How to Disturb Network Reconnaissance" paper

set -e  # Exit on error

echo "======================================================================"
echo "ID-HAM Artifact Evaluation - Environment Setup"
echo "======================================================================"
echo ""

# Parse command line arguments
SKIP_MININET=false
SKIP_UPDATE=false
USE_TF2=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-mininet)
            SKIP_MININET=true
            shift
            ;;
        --skip-update)
            SKIP_UPDATE=true
            shift
            ;;
        --use-tf2)
            USE_TF2=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-mininet    Skip Mininet installation (if already installed)"
            echo "  --skip-update     Skip apt-get update"
            echo "  --use-tf2         Use TensorFlow 2.x instead of 1.14"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check Python version
echo "[1/6] Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Found Python $PYTHON_VERSION"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,7) else 1)"; then
    echo "  ERROR: Python 3.7 or higher required"
    exit 1
fi
echo "  ✓ Python version OK"

# Update system
if [ "$SKIP_UPDATE" = false ]; then
    echo ""
    echo "[2/6] Updating package lists..."
    sudo apt-get update -qq
    echo "  ✓ Package lists updated"
else
    echo ""
    echo "[2/6] Skipping system update (--skip-update)"
fi

# Install Mininet (optional)
echo ""
if [ "$SKIP_MININET" = false ]; then
    echo "[3/6] Installing Mininet..."
    echo "  Note: This may take 10-15 minutes"
    
    if command -v mn &> /dev/null; then
        echo "  Mininet already installed, skipping..."
    else
        if [ ! -d "mininet" ]; then
            git clone https://github.com/mininet/mininet
            cd mininet
            git checkout 2.3.0
            sudo PYTHON=python3 util/install.sh -a
            cd ..
        else
            echo "  Mininet directory exists, skipping download..."
        fi
    fi
    echo "  ✓ Mininet ready"
else
    echo "[3/6] Skipping Mininet installation (--skip-mininet)"
fi

# Install Ryu SDN Controller
echo ""
echo "[4/6] Installing Ryu SDN Controller..."
pip install ryu==4.34 --break-system-packages --quiet
echo "  ✓ Ryu Controller installed"

# Install Python packages
echo ""
echo "[5/6] Installing Python dependencies..."

# TensorFlow
if [ "$USE_TF2" = true ]; then
    echo "  Installing TensorFlow 2.x..."
    pip install tensorflow==2.9 --break-system-packages --quiet
else
    echo "  Installing TensorFlow 1.14 (may show warnings)..."
    pip install tensorflow==1.14 --break-system-packages --quiet 2>/dev/null || \
    pip install tensorflow==1.15 --break-system-packages --quiet || \
    (echo "  TensorFlow 1.x failed, falling back to 2.x..." && \
     pip install tensorflow==2.9 --break-system-packages --quiet)
fi

# Other packages
echo "  Installing NetworkX..."
pip install networkx==2.5 --break-system-packages --quiet

echo "  Installing NumPy and Matplotlib..."
pip install numpy matplotlib --break-system-packages --quiet

echo "  Installing Z3 Theorem Prover..."
pip install z3-solver --break-system-packages --quiet

echo "  ✓ All Python packages installed"

# Verify installations
echo ""
echo "[6/6] Verifying installations..."

# Check TensorFlow
if python3 -c "import tensorflow as tf; print(f'  ✓ TensorFlow {tf.__version__}')" 2>/dev/null; then
    :
else
    echo "  ✗ TensorFlow import failed"
    exit 1
fi

# Check NetworkX
if python3 -c "import networkx as nx; print(f'  ✓ NetworkX {nx.__version__}')" 2>/dev/null; then
    :
else
    echo "  ✗ NetworkX import failed"
    exit 1
fi

# Check Z3
if python3 -c "import z3; print('  ✓ Z3 Solver OK')" 2>/dev/null; then
    :
else
    echo "  ✗ Z3 import failed"
    exit 1
fi

# Check NumPy
if python3 -c "import numpy as np; print(f'  ✓ NumPy {np.__version__}')" 2>/dev/null; then
    :
else
    echo "  ✗ NumPy import failed"
    exit 1
fi

# Check Matplotlib
if python3 -c "import matplotlib; print('  ✓ Matplotlib OK')" 2>/dev/null; then
    :
else
    echo "  ✗ Matplotlib import failed"
    exit 1
fi

# Create output directories
echo ""
echo "Creating output directories..."
mkdir -p results
mkdir -p checkpoints
mkdir -p logs
echo "  ✓ Directories created"

# Summary
echo ""
echo "======================================================================"
echo "Setup Complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Quick test (10-15 min):  python run_experiments_quick.py"
echo "  2. Full experiments (6-10h): python run_experiments.py"
echo "  3. SMT evaluation (30 min):  python evaluate_smt_performance.py"
echo ""
echo "Documentation:"
echo "  - GETTING_STARTED.md: Quick start guide"
echo "  - README.md: Comprehensive documentation"
echo "  - ARTIFACT_EVALUATION.md: Detailed validation procedures"
echo ""
echo "======================================================================"
