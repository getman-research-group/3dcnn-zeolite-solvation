#!/bin/bash

# --- SLURM Resource Request ---
#SBATCH --job-name=pytorch_benchmark                    # Job name
#SBATCH --account=PAS2536                               # Your account name
#SBATCH --cluster=ascend                                # Run on the Ascend cluster
#SBATCH --nodes=1                                       # Request a single node
#SBATCH --ntasks-per-node=8                             # Request 8 CPU cores
#SBATCH --gpus-per-node=1                               # Request 1 GPU
#SBATCH --time=00:10:00                                 # Job runtime (10 minutes)
#SBATCH --output=../output_model_cnn/benchmark_%j.out   # Standard output and error log

# --- Job Steps ---

# 1. Create a directory for log files, if it doesn't exist
mkdir -p ../output_model_cnn

# 2. Print job context
echo "Date:              $(date)"
echo "Job ID:            $SLURM_JOB_ID"
echo "Job Name:          $SLURM_JOB_NAME"
echo "Node(s) running on: $SLURM_JOB_NODELIST"
echo "Current directory: $(pwd)"
echo "-------------------------------------------------"

# 3. Set up the software environment
echo "Activating Conda environment..."
# Using your specified environment name: 'torch'
conda activate torch
echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "-------------------------------------------------"

# 4. Execute the Python benchmark script
echo "Running the Python benchmark script..."
# Using your specified Python script name: 'test_pytorch.py'
python test_pytorch.py --size 4096

echo "-------------------------------------------------"
echo "Job finished successfully."