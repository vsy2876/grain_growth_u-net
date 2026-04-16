#!/bin/bash
#SBATCH --job-name=GrainDiff_Train
#SBATCH --account=mse
#SBATCH --qos=coe-ice
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00 
#SBATCH --output=./logs/grain_train_%j.out 
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=vyadav68@gatech.edu

# Print minimal job info
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"
echo "Training Grain Growth Conditional Diffusion Model on H200"
echo ""

# Ensure the logs directory exists
mkdir -p ./logs

# Move to submission directory
cd /home/hice1/vyadav68/scratch/grain_growth || exit 1

# Load the working system compiler
module load gcc
module load cuda/12.6.1
export LD_LIBRARY_PATH=$CUDA_HOME/lib64/stubs:/usr/lib64:$LD_LIBRARY_PATH

# Load environment
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate polymers

# Set environment variables for H200 optimization
export CUDA_LAUNCH_BLOCKING=0      # Async GPU ops (faster)
export TORCH_CUDNN_BENCHMARK=1     # Auto-tune cuDNN (faster after warmup)
export OMP_NUM_THREADS=12          # Match NUM_WORKERS=12 in train.py
export MKL_NUM_THREADS=12          # Math library threads (match OMP)

export TORCHINDUCTOR_CACHE_DIR=/home/hice1/vyadav68/scratch/grain_growth/.torch_cache
mkdir -p $TORCHINDUCTOR_CACHE_DIR

# Quick GPU check
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Run training using the absolute path
echo "Starting training..."
cd /home/hice1/vyadav68/scratch/grain_growth/diff_multi_frame
python3 /home/hice1/vyadav68/scratch/grain_growth/diff_multi_frame/last_diff_train.py

# Capture exit code
EXIT_CODE=$?

# Print completion status
echo ""
echo "Training finished with exit code: $EXIT_CODE | End: $(date)"

exit $EXIT_CODE