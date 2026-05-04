#!/bin/bash

#SBATCH --account=bienkows_1550
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=flatten-data
#SBATCH --output=/project2/bienkows_1550/ee541_proj/flatten_%j.out
#SBATCH --error=/project2/bienkows_1550/ee541_proj/flatten_%j.err

echo "=========================================="
echo "Flattening CHB-MIT Directory Structure"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "=========================================="

cd $SLURM_SUBMIT_DIR

PHYSIONET_DIR="data/raw/physionet.org/files/chbmit/1.0.0"
DATA_DIR="data/raw"

if [ ! -d "$PHYSIONET_DIR" ]; then
    echo "ERROR: PhysioNet directory not found: $PHYSIONET_DIR"
    exit 1
fi

echo ""
echo "Source: $PHYSIONET_DIR"
echo "Destination: $DATA_DIR"
echo ""

# Copy each chb* directory to root level
count=0
for dir in "$PHYSIONET_DIR"/chb*/; do
    if [ -d "$dir" ]; then
        subject=$(basename "$dir")
        dest="$DATA_DIR/$subject"
        
        if [ -d "$dest" ]; then
            echo "⚠ $subject already exists, skipping..."
        else
            echo "Copying $subject..."
            cp -r "$dir" "$dest"
            count=$((count + 1))
        fi
    fi
done

echo ""
echo "=========================================="
echo "Flattening Complete"
echo "Subjects copied: $count"
echo "End time: $(date)"
echo "=========================================="
