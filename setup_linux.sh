#!/bin/bash
# ML HW5 Linux 環境設置腳本
# Author: Enhanced implementation
# Date: 2025-11-26

set -e  # Exit on error

echo "========================================================================"
echo "ML HW5 - Linux Environment Setup"
echo "========================================================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found. Please install Miniconda or Anaconda first.${NC}"
    exit 1
fi

echo -e "\n${GREEN}[Step 1/6]${NC} Cleaning up macOS-specific files..."
if [ -f "libsvm_wrapper.py" ]; then
    mv libsvm_wrapper.py libsvm_wrapper.py.macos_backup
    echo "  ✓ Backed up libsvm_wrapper.py"
fi

if [ -d "libsvm" ]; then
    mv libsvm libsvm.macos_backup
    echo "  ✓ Backed up libsvm/ directory"
fi

echo -e "\n${GREEN}[Step 2/6]${NC} Creating conda environment..."
if conda env list | grep -q "ml_hw5"; then
    echo -e "${YELLOW}  Environment 'ml_hw5' already exists. Skipping creation.${NC}"
else
    conda create -n ml_hw5 python=3.8 -y
    echo "  ✓ Created environment 'ml_hw5'"
fi

echo -e "\n${GREEN}[Step 3/6]${NC} Activating environment..."
# Note: This needs to be run in the parent shell, so we'll provide instructions
echo -e "${YELLOW}  Please run: conda activate ml_hw5${NC}"

echo -e "\n${GREEN}[Step 4/6]${NC} Installing Python packages..."
echo "  Installing numpy==1.19.0..."
pip install numpy==1.19.0 -q

echo "  Installing scipy==1.5.1..."
pip install scipy==1.5.1 -q

echo "  Installing pandas==1.1.5..."
pip install pandas==1.1.5 -q

echo "  Installing matplotlib..."
pip install matplotlib -q

echo "  Installing libsvm-official..."
pip install -U libsvm-official

echo -e "\n${GREEN}[Step 5/6]${NC} Verifying installation..."
python << 'PYEOF'
import sys
try:
    import numpy as np
    import scipy
    import pandas as pd
    import matplotlib.pyplot as plt
    from libsvm.svmutil import *

    print(f"  ✓ NumPy: {np.__version__}")
    print(f"  ✓ SciPy: {scipy.__version__}")
    print(f"  ✓ Pandas: {pd.__version__}")
    print(f"  ✓ libsvm: imported successfully")
    print(f"\n  All packages installed correctly!")
except ImportError as e:
    print(f"  ✗ Error: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo -e "${RED}Installation verification failed!${NC}"
    exit 1
fi

echo -e "\n${GREEN}[Step 6/6]${NC} Running quick test..."
python test_svm_quick.py

echo -e "\n========================================================================"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Run Gaussian Process:  python gaussian_process.py"
echo "  2. Run SVM (full):        python svm_mnist.py"
echo "  3. Run all tasks:         python run_all.py"
echo ""
echo "To save output:"
echo "  python svm_mnist.py > svm_output_linux.txt 2>&1"
echo ""
