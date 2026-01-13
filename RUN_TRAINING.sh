#!/bin/bash
#
# Quick Start: Run Model Training (Phase 5)
#
# This script trains all 3 models with the fixed feature engineering.
# Expected time: 15-30 minutes
# Expected output: 3 model files in models/rebuilt_from_scratch/
#

set -e  # Exit on error

echo "================================================================================"
echo "Phase 5: Training Models with Fixed Features"
echo "================================================================================"
echo ""
echo "Starting: $(date)"
echo ""

# Check we're in the right directory
if [ ! -f "rebuild_from_scratch.py" ]; then
    echo "ERROR: Must run from project root (/Users/omar/Desktop/ML/xauusd_signals)"
    exit 1
fi

# Check data file exists
if [ ! -f "data/features/xauusd_features_2020_2025_fixed.parquet" ]; then
    echo "ERROR: Training data file not found!"
    echo "Expected: data/features/xauusd_features_2020_2025_fixed.parquet"
    exit 1
fi

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import numpy, pandas, sklearn, joblib" 2>/dev/null || {
    echo "ERROR: Missing Python dependencies!"
    echo "Install: pip install numpy pandas scikit-learn joblib pyarrow"
    exit 1
}
echo "✓ Dependencies OK"
echo ""

# Create output directory
mkdir -p models/rebuilt_from_scratch
echo "✓ Output directory ready"
echo ""

# Run training
echo "Starting training pipeline..."
echo "This will take 15-30 minutes. Progress will be shown below."
echo ""
echo "================================================================================"
echo ""

python3 rebuild_from_scratch.py 2>&1 | tee training_log_$(date +%Y%m%d_%H%M%S).txt

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✅ TRAINING COMPLETE!"
    echo "================================================================================"
    echo ""
    echo "Models saved:"
    ls -lh models/rebuilt_from_scratch/*.joblib 2>/dev/null || echo "⚠️ No models found!"
    echo ""
    echo "Training log saved: training_log_*.txt"
    echo ""
    echo "Next steps:"
    echo "  1. Review training results in the log"
    echo "  2. Verify win rates ≥52% (target 65-70%)"
    echo "  3. If successful, proceed to Phase 6 (deployment)"
    echo ""
else
    echo ""
    echo "================================================================================"
    echo "❌ TRAINING FAILED!"
    echo "================================================================================"
    echo ""
    echo "Check the log above for errors."
    echo "Common issues:"
    echo "  - Data file corrupted or missing columns"
    echo "  - Memory errors (need more RAM)"
    echo "  - Module import errors (check PYTHONPATH)"
    echo ""
    echo "See PHASE5_TRAINING_INSTRUCTIONS.md for troubleshooting."
    echo ""
    exit 1
fi
