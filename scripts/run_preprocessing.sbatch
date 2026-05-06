#!/bin/bash

#SBATCH --account=bienkows_1550
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --job-name=eeg-preprocess
#SBATCH --output=/project2/bienkows_1550/ee541_proj/preprocessing_%j.out
#SBATCH --error=/project2/bienkows_1550/ee541_proj/preprocessing_%j.err

echo "=========================================="
echo "Starting EEG Preprocessing Pipeline"
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

echo ""
echo "Running preprocessing pipeline..."
echo ""

time python -u src/preprocessing/pipeline.py \
    --data_dir data/raw/physionet.org/files/chbmit/1.0.0 \
    --out_dir /project2/bienkows_1550/ee541_proj/processed \
    --max_windows 25000

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job Summary"
echo "Job ID: $SLURM_JOB_ID"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
