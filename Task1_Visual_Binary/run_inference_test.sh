#!/bin/bash
# Quick Test Script for Judges to Verify Task 1 Model

echo "========================================"
echo "TASK 1: INFERENCE VERIFICATION"
echo "========================================"
echo ""
echo "Running inference on all test videos..."
echo ""

cd Task1_Visual_Binary
source ../venv/bin/activate
python batch_inference.py "../videos for testing/"

echo ""
echo "========================================"
echo "✅ VERIFICATION COMPLETE"
echo "========================================"
echo ""
echo "See INFERENCE_RESULTS.md for detailed results"
echo ""
