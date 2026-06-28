#!/bin/bash

# --- SLURM Resource Request ---
#SBATCH --job-name=3d_cnn                       # specific job name
#SBATCH --account=PAS2536                       # Your project account
#SBATCH --cluster=ascend                        # Explicitly request the Ascend cluster for A100 GPUs
#SBATCH --nodes=1                               # Request one whole node
#SBATCH --ntasks-per-node=8                     # Reduced CPU cores (more reasonable for GPU job)
#SBATCH --gpus-per-node=1                       # Request 1 NVIDIA A100 GPU
#SBATCH --mem=128G                              # 128GB RAM for larger model and dataset
#SBATCH --time=20:00:00                         # Extended time for 3D CNN model training
#SBATCH --output=/dev/null                      # Disable default SLURM output file (slurm-jobid.out)
#SBATCH --error=/dev/null                       # Disable default SLURM error file (slurm-jobid.err)


# --- Job Logic ---

# 1. SET UP JOB PARAMETERS
# Parse both positional and named parameters
# Default values - Updated for CNN model training
MODE="full"
TEST_MODE="false"
MODEL_FILE="train_3d_cnn"    # Default model file to execute (updated version with split-type support)
EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=0.001
RANDOM_STATE=42
SPLIT_TYPE="random_split"

# Parse arguments
for arg in "$@"; do
    case $arg in
        model-file=*)
            MODEL_FILE="${arg#*=}"
            ;;
        test-mode=*)
            TEST_MODE="${arg#*=}"
            if [ "$TEST_MODE" = "true" ]; then
                MODE="test"
            else
                MODE="full"
            fi
            ;;
        epochs=*)
            EPOCHS="${arg#*=}"
            ;;
        batch-size=*)
            BATCH_SIZE="${arg#*=}"
            ;;
        learning-rate=*)
            LEARNING_RATE="${arg#*=}"
            ;;
        random-state=*)
            RANDOM_STATE="${arg#*=}"
            ;;
        split-type=*)
            SPLIT_TYPE="${arg#*=}"
            ;;
        test|full|retrain|quick|debug|custom)
            MODE="$arg"
            ;;
        *)
            # Handle positional arguments for backward compatibility
            if [ -z "$MODE_SET" ]; then
                MODE="$arg"
                MODE_SET=true
            elif [ -z "$EPOCHS_SET" ]; then
                EPOCHS="$arg"
                EPOCHS_SET=true
            elif [ -z "$BATCH_SIZE_SET" ]; then
                BATCH_SIZE="$arg"
                BATCH_SIZE_SET=true
            elif [ -z "$LEARNING_RATE_SET" ]; then
                LEARNING_RATE="$arg"
                LEARNING_RATE_SET=true
            elif [ -z "$RANDOM_STATE_SET" ]; then
                RANDOM_STATE="$arg"
                RANDOM_STATE_SET=true
            fi
            ;;
    esac
done

# Extract model name from model file for naming purposes
# Convert train_3d_cnn -> model, etc.
if [[ "$MODEL_FILE" =~ train_3d_cnn_(.+)$ ]]; then
    MODEL_NAME="model_${BASH_REMATCH[1]}"
else
    # Fallback: use the model file name directly if pattern doesn't match
    MODEL_NAME="$MODEL_FILE"
fi

# Create split type suffix for file naming
SPLIT_SUFFIX=""
if [ "$SPLIT_TYPE" = "random_split" ]; then
    SPLIT_SUFFIX="random"
elif [ "$SPLIT_TYPE" = "solvent_split" ]; then
    SPLIT_SUFFIX="solvent"
elif [ "$SPLIT_TYPE" = "pore_type_split" ]; then
    SPLIT_SUFFIX="pore_type"
fi

# Create logs directory if it doesn't exist to store the output file.
mkdir -p ../output_model_cnn

# Dynamically set output file names based on model name (extracted from model-file) and start redirection
# This needs to be done after parsing MODEL_FILE and extracting MODEL_NAME
if [ "$SLURM_JOB_ID" ]; then
    # Define the final log file names based on model name - split type before SLURM job ID
    OUTPUT_LOG="../output_model_cnn/${MODEL_NAME}-${SPLIT_SUFFIX}-${SLURM_JOB_ID}.log"
    ERROR_LOG="../output_model_cnn/${MODEL_NAME}-${SPLIT_SUFFIX}-${SLURM_JOB_ID}.err"
    
    echo "Setting up model-specific log files..."
    echo "  - Model File: $MODEL_FILE"
    echo "  - Model Name: $MODEL_NAME"
    echo "  - Output: $OUTPUT_LOG"
    echo "  - Error:  $ERROR_LOG"
    
    # Redirect all subsequent output to the model-specific files
    exec 1> "$OUTPUT_LOG" 2> "$ERROR_LOG"
    
    # Also echo to the console initially
    echo "Output redirection started to ${MODEL_NAME}-specific files" | tee -a "$OUTPUT_LOG"
else
    echo "Warning: SLURM_JOB_ID not found, using console output only"
fi

# Print parameter information for debugging
echo "================================================================="
echo "  3D CNN MODEL TRAINING CONFIGURATION"
echo "================================================================="
echo "Command used: sbatch train_3d_cnn.sh $@"
echo "Parsed parameters:"
echo "  - MODEL_FILE:       $MODEL_FILE"
echo "  - MODEL_NAME:       $MODEL_NAME (for file naming)"
echo "  - MODE:             $MODE"
echo "  - TEST_MODE:        $TEST_MODE"
echo "  - EPOCHS:           $EPOCHS"
echo "  - BATCH_SIZE:       $BATCH_SIZE"
echo "  - LEARNING_RATE:    $LEARNING_RATE"
echo "  - RANDOM_STATE:     $RANDOM_STATE"
echo "  - SPLIT_TYPE:       $SPLIT_TYPE"
echo "================================================================="


# 2. PRINT JOB INFORMATION
# This information is invaluable for debugging and keeping records.
echo "================================================================="
echo "  STARTING 3D CNN MODEL TRAINING JOB"
echo "================================================================="
echo "Job Name:           $SLURM_JOB_NAME"
echo "Job ID:             $SLURM_JOB_ID"
echo "Execution Time:     $(date)"
echo "Running on Node:    $(hostname)"
echo "Requested GPU(s):   $SLURM_GPUS_ON_NODE"
echo "Model File:         $MODEL_FILE.py"
echo "Model Name:         $MODEL_NAME"
echo "Training Mode:      $MODE"
echo "Epochs:             $EPOCHS"
echo "Batch Size:         $BATCH_SIZE"
echo "Learning Rate:      $LEARNING_RATE"
echo "Random State:       $RANDOM_STATE"
echo "-----------------------------------------------------------------"


# 3. SET UP THE SOFTWARE ENVIRONMENT
# First, source the .bashrc to make conda functions available to the script
echo "Sourcing .bashrc to initialize shell for Conda..."
source ~/.bashrc

echo "Activating self-contained Conda environment: torch"
# Add error checking for conda activation
if ! conda activate torch; then
    echo "Error: Failed to activate conda environment 'torch'"
    exit 1
fi

echo "Environment Details:"
echo "  - Python Path: $(which python)"
echo "  - Conda Env:   $CONDA_DEFAULT_ENV"
echo "  - PyTorch:     $(python -c 'import torch; print(torch.__version__)')"
echo "  - CUDA Ready:  $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  - GPU Count:   $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "  - GPU Name:    $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "-----------------------------------------------------------------"


# 4. SET WORKING DIRECTORY TO PROJECT LOCATION
#
# Running directly in the project directory on ESS storage
#
echo "Setting up working directory in project location..."
WORK_DIR="${SLURM_SUBMIT_DIR}"
cd "$WORK_DIR"

echo "  - Work Directory: $WORK_DIR"
echo "  - Current directory: $(pwd)"
echo "  - Available files: $(ls -la | wc -l) files"

# Verify essential files exist
if [ ! -f "${MODEL_FILE}.py" ]; then
    echo "Error: ${MODEL_FILE}.py not found in current directory"
    echo "Current directory contents:"
    ls -la
    exit 1
fi

if [ ! -d "core" ]; then
    echo "Error: core directory not found"
    exit 1
fi

echo "  - Essential files verified (${MODEL_FILE}.py exists)"
echo "-----------------------------------------------------------------"


# 5. EXECUTE THE PYTHON TRAINING SCRIPT
echo "Starting Python training script: ${MODEL_FILE}.py..."
case $MODE in
    "test")
        python -u ${MODEL_FILE}.py --test-mode --verbose --random-state $RANDOM_STATE --split-type $SPLIT_TYPE --output-prefix $MODEL_NAME
        ;;
    "full")
        python -u ${MODEL_FILE}.py --verbose --epochs $EPOCHS --batch-size $BATCH_SIZE --learning-rate $LEARNING_RATE --random-state $RANDOM_STATE --split-type $SPLIT_TYPE --output-prefix $MODEL_NAME
        ;;
    "retrain")
        python -u ${MODEL_FILE}.py --retrain --verbose --epochs $EPOCHS --batch-size $BATCH_SIZE --learning-rate $LEARNING_RATE --random-state $RANDOM_STATE --split-type $SPLIT_TYPE --output-prefix $MODEL_NAME
        ;;
    "quick")
        python -u ${MODEL_FILE}.py --test-mode --verbose --random-state $RANDOM_STATE --split-type $SPLIT_TYPE --output-prefix $MODEL_NAME
        ;;
    "debug")
        python -u ${MODEL_FILE}.py --test-mode --verbose --split-type random_split --random-state $RANDOM_STATE --output-prefix $MODEL_NAME
        ;;
    "custom")
        # For custom mode, use all provided parameters
        python -u ${MODEL_FILE}.py --verbose --epochs $EPOCHS --batch-size $BATCH_SIZE --learning-rate $LEARNING_RATE --random-state $RANDOM_STATE --split-type $SPLIT_TYPE --output-prefix $MODEL_NAME
        ;;
    *)
        echo "Available modes:"
        echo "  test     - Test mode with limited data and epochs"
        echo "  full     - Full training mode (default)"
        echo "  retrain  - Force retrain existing models"
        echo "  quick    - Very quick test (same as test mode)"
        echo "  debug    - Debug mode with 2 CV folds"
        echo "  custom   - Custom training with specified parameters"
        echo ""
        echo "Model File Selection:"
        echo "  model-file=train_3d_cnn - Use train_3d_cnn.py training script (default, cleaned CNN)"
        echo ""
        echo "Usage examples:"
        echo "  Positional: sbatch train_3d_cnn.sh [MODE] [EPOCHS] [BATCH_SIZE] [LEARNING_RATE] [RANDOM_STATE]"
        echo "  Example: sbatch train_3d_cnn.sh full 100 32 0.001 42"
        echo ""
        echo "  Named parameters: sbatch train_3d_cnn.sh [param=value] ..."
        echo "  Example: sbatch train_3d_cnn.sh model-file=train_3d_cnn epochs=100 batch-size=16 split-type=solvent_split"
        echo ""
        echo "Split-type examples:"
        echo "  random_split (5-fold):      sbatch train_3d_cnn.sh full split-type=random_split"
        echo "  solvent_split (4-fold):     sbatch train_3d_cnn.sh full split-type=solvent_split"  
        echo "  pore_type_split (2-fold):   sbatch train_3d_cnn.sh full split-type=pore_type_split"
        echo ""
        echo "Available named parameters:"
        echo "  model-file=FILENAME       - Python training script to execute (without .py extension)"
        echo "  test-mode=true/false      - Enable/disable test mode"
        echo "  epochs=N                  - Number of training epochs"
        echo "  batch-size=N              - Training batch size"
        echo "  learning-rate=X           - Learning rate for optimizer"
        echo "  random-state=N            - Random seed for reproducibility"
        echo "  split-type=TYPE           - Cross-validation split strategy: random_split (5-fold), solvent_split (4-fold), pore_type_split (2-fold)"
        echo "Error: Unknown mode '$MODE'. Exiting."
        exit 1
        ;;
esac

# Capture the exit code of the Python script to determine job success.
PYTHON_EXIT_CODE=$?
echo "Python script finished with exit code: $PYTHON_EXIT_CODE"
echo "-----------------------------------------------------------------"


# 6. TRAINING COMPLETED - RESULTS ARE ALREADY IN PROJECT DIRECTORY
#
# Since we're running directly in the project directory, no file copying needed
#
echo "Model training completed. All results are saved in the output directory."
echo ""
echo "Actual file locations on server:"
echo "  - Output Directory: /fs/ess/PAS2536/jiexins/zeolite_project/output_model_cnn/"
echo "  - PKL, PTH, and log files are all in the above directory."
echo "  - Files will be prefixed with: ${MODEL_NAME}"


# Monitor GPU usage during training
echo ""
echo "GPU Usage Summary:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
else
    echo "nvidia-smi not available"
fi


# 7. FINAL JOB STATUS
echo ""
echo "================================================================="
echo "  3D CNN MODEL TRAINING JOB COMPLETED"
echo "  Model File: $MODEL_FILE.py"
echo "  Model Name: $MODEL_NAME"
echo "  Completion Time: $(date)"
echo "  Final Status (based on Python exit code): $PYTHON_EXIT_CODE"
echo "================================================================="

# Exit with the same code as the Python script, so Slurm knows if the job succeeded.
exit $PYTHON_EXIT_CODE

# =================================================================================
# 3D CNN MODEL TRAINING USAGE EXAMPLES FOR CLUSTER SUBMISSION
# =================================================================================
#
# 1. Default CNN Model (train_3d_cnn) - Full Dataset Training:
#    sbatch train_3d_cnn.sh model-file=train_3d_cnn epochs=100 batch-size=32 learning-rate=0.001
#    Output files: model_JOBID.log, model_JOBID.err, model_JOBID_*.pth, model_JOBID_*.pkl
#
# 2. Quick Test Mode:
#    sbatch train_3d_cnn.sh test
#
# 3. Debug Mode (2 CV folds only):
#    sbatch train_3d_cnn.sh debug
#
# 4. Custom Parameters:
#    sbatch train_3d_cnn.sh epochs=50 batch-size=16 learning-rate=0.0005
#
# 5. Using Different Model File:
#    sbatch train_3d_cnn.sh model-file=train_3d_cnn_custom epochs=100
