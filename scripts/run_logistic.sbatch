#!/bin/bash

#SBATCH --account=bienkows_1550
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=logistic
#SBATCH --output=/project2/bienkows_1550/ee541_proj/logistic_%j.out
#SBATCH --error=/project2/bienkows_1550/ee541_proj/logistic_%j.err

echo "=========================================="
echo "Architecture 1: Logistic Regression Baseline"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "=========================================="

cd $SLURM_SUBMIT_DIR

source /home1/bbrening/.bashrc
conda activate ee541
echo "Python: $(which python)"

echo ""
time python -u src/training/evaluate.py \
    --arch logistic \
    --processed_dir /project2/bienkows_1550/ee541_proj/processed \
    --results_dir /project2/bienkows_1550/ee541_proj/results

echo ""
echo "Exit code: $?"
echo "End time: $(date)"
