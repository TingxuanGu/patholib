#!/bin/bash
# =============================================================
# patholib deployment script for GPU server
# =============================================================
# Usage:
#   chmod +x setup.sh && ./setup.sh [--gpu] [--wsi]
#
# Options:
#   --gpu   Install PyTorch with CUDA + Cellpose (requires NVIDIA GPU)
#   --wsi   Install OpenSlide for WSI support
# =============================================================

set -e

echo "=== patholib setup ==="

# Parse flags
GPU=false
WSI=false
for arg in "$@"; do
    case $arg in
        --gpu) GPU=true ;;
        --wsi) WSI=true ;;
    esac
done

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
echo "Installing core dependencies..."
pip install numpy scipy scikit-image opencv-python-headless Pillow matplotlib

# Optional: GPU (PyTorch + Cellpose)
if [ "$GPU" = true ]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    echo "Installing Cellpose..."
    pip install cellpose
fi

# Optional: WSI (OpenSlide)
if [ "$WSI" = true ]; then
    echo "Installing OpenSlide Python bindings..."
    pip install openslide-python
    echo ""
    echo "*** IMPORTANT: You also need system-level OpenSlide ***"
    echo "  Ubuntu/Debian: sudo apt install openslide-tools"
    echo "  CentOS/RHEL:   sudo yum install openslide"
    echo "  macOS:         brew install openslide"
fi

# Install patholib as editable package
echo "Installing patholib..."
pip install -e .

echo ""
echo "=== Setup complete ==="
echo "Activate with: source venv/bin/activate"
echo ""
echo "Quick test:"
echo "  python analyze_ihc.py --help"
echo "  python analyze_he.py --help"
