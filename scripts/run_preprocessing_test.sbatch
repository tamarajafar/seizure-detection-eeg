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

cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Activate conda environment
source /home1/bbrening/.bashrc
conda activate ee541
echo "Python: $(which python)"

# Create a test data directory with symbolic links to just chb01-05
TEST_DATA_DIR="data/test_raw"
rm -rf "$TEST_DATA_DIR"  # Clean up any stale symlinks
mkdir -p "$TEST_DATA_DIR"

echo ""
echo "Setting up test data with chb01-chb05..."

# Get absolute paths
PHYSIONET_PATH="$(pwd)/data/raw/physionet.org/files/chbmit/1.0.0"
FLAT_PATH="$(pwd)/data/raw"

if [ -d "$PHYSIONET_PATH/chb01" ]; then
    echo "Found nested PhysioNet structure"
    BASE_PATH="$PHYSIONET_PATH"
elif [ -d "$FLAT_PATH/chb01" ]; then
    echo "Found flat directory structure"
    BASE_PATH="$FLAT_PATH"
else
    echo "ERROR: Could not find chb01 in either nested or flat structure"
    exit 1
fi

for i in {01..05}; do
    subject="chb$i"
    source_path="$BASE_PATH/$subject"
    if [ -d "$source_path" ]; then
        ln -sf "$source_path" "$TEST_DATA_DIR/$subject"
        echo "  Linked $subject"
    else
        echo "  WARNING: $subject not found at $source_path"
    fi
done

echo ""
echo "Running preprocessing on test data..."
echo ""

time python -u src/preprocessing/pipeline.py \
    --data_dir "$TEST_DATA_DIR" \
    --out_dir /project2/bienkows_1550/ee541_proj/processed_test

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
echo "Output saved to: /project2/bienkows_1550/ee541_proj/processed_test/"
echo ""

exit $EXIT_CODE
