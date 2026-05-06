#!/bin/bash

#SBATCH --account=bienkows_1550
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --job-name=cnn-cross
#SBATCH --output=/project2/bienkows_1550/ee541_proj/cnn_cross_%j.out
#SBATCH --error=/project2/bienkows_1550/ee541_proj/cnn_cross_%j.err

echo "=========================================="
echo "Architecture 3: CNN-LSTM (Cross-Subject)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

cd $SLURM_SUBMIT_DIR

source /home1/bbrening/.bashrc
conda activate ee541
echo "Python: $(which python)"

echo ""
time python -u src/training/evaluate.py \
    --arch cnn_lstm \
    --processed_dir /project2/bienkows_1550/ee541_proj/processed \
    --results_dir /project2/bienkows_1550/ee541_proj/results

echo ""
echo "Exit code: $?"
echo "End time: $(date)"
