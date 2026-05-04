#!/bin/bash

#SBATCH --account=bienkows_1550
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=inspect-data
#SBATCH --output=/project2/bienkows_1550/ee541_proj/inspect_%j.out
#SBATCH --error=/project2/bienkows_1550/ee541_proj/inspect_%j.err

echo "=========================================="
echo "Data Inspection Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "=========================================="

cd $SLURM_SUBMIT_DIR

python scripts/data_inspector.py data/raw

echo ""
echo "=========================================="
echo "Deep Directory Structure (5 levels)"
echo "=========================================="
tree -L 5 -h data/raw 2>/dev/null || find data/raw -maxdepth 5 -type d | head -100

echo ""
echo "=========================================="
echo "Sample files in each subdirectory"
echo "=========================================="
find data/raw -maxdepth 5 -type f | head -50

echo ""
echo "=========================================="
echo "Data Inspection Completed: $(date)"
echo "=========================================="
