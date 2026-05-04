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

echo ""
echo "Running preprocessing on chb01-chb05..."
echo ""

time python -u src/preprocessing/pipeline.py \
    --data_dir data/raw/physionet.org/files/chbmit/1.0.0 \
    --out_dir /project2/bienkows_1550/ee541_proj/processed_test \
    --subjects chb01 chb02 chb03 chb04 chb05

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
