#!/bin/bash

#SBATCH --account=bienkows_1550
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --job-name=preproc-test
#SBATCH --output=/project2/bienkows_1550/ee541_proj/preproc_test_%j.out
#SBATCH --error=/project2/bienkows_1550/ee541_proj/preproc_test_%j.err

echo "=========================================="
echo "EEG Preprocessing - TEST (chb01-05 only)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "=========================================="

# Create output directory if it doesn't exist
mkdir -p /project2/bienkows_1550/ee541_proj

# Get repo root (parent of scripts directory)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
cd "$REPO_ROOT"

echo "Working directory: $(pwd)"

# Create a test data directory with symbolic links to just chb01-05
TEST_DATA_DIR="data/test_raw"
mkdir -p "$TEST_DATA_DIR"

echo ""
echo "Setting up test data with chb01-chb05..."
for i in {01..05}; do
    subject="chb$i"
    if [ -d "data/raw/$subject" ]; then
        ln -sf "../../raw/$subject" "$TEST_DATA_DIR/$subject"
        echo "  Linked $subject"
    fi
done

echo ""
echo "Running preprocessing on test data..."
echo ""

time python -u src/preprocessing/pipeline.py \
    --data_dir "$TEST_DATA_DIR" \
    --out_dir data/processed_test

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Test Preprocessing Complete"
echo "Job ID: $SLURM_JOB_ID"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "To scale up: multiply time by 24/5 = 4.8x"
echo "Output saved to: data/processed_test/"
echo ""

exit $EXIT_CODE
