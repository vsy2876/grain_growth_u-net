#!/bin/bash
#SBATCH --job-name=GrainGrowth_Gen
#SBATCH --account=mse                    # Your PACE account
#SBATCH --qos=coe-ice                    # Your specific QoS
#SBATCH --nodes=1 
#SBATCH --ntasks=4                       # 4 cores PER simulation (Matches your Python script)
#SBATCH --mem-per-cpu=2G                 # 8GB total per simulation
#SBATCH --time=00:45:00                  # Max time for ONE simulation (45 mins is plenty)
#SBATCH --array=501-1000               # <--- THIS SPAWNS 500 JOBS, (Max Limit 500, hence run twice for 1000 total)
#SBATCH --output=./logs/gen_%A_%a.out    # %A=Job ID, %a=Array ID
#SBATCH --error=./logs/gen_%A_%a.err     

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Print job info
echo "Array Task ID: $SLURM_ARRAY_TASK_ID | Node: $SLURM_NODELIST | Start: $(date)"

# Move to submission directory
cd $SLURM_SUBMIT_DIR || exit 1

# Load exact modules used to compile SPPARKS
module purge
module load anaconda3          
module load gcc/12.3.0
module load openmpi/4.1.5

# Activate your environment so python has scipy and numpy
conda activate polymers

# Run the python data generation script
# It will automatically pick up the SLURM_ARRAY_TASK_ID
echo "Starting data generation for run $SLURM_ARRAY_TASK_ID..."
python3 grain_dataset.py

# Capture exit code
EXIT_CODE=$?

# Print completion status
echo "Data generation finished with exit code: $EXIT_CODE | End: $(date)"

exit $EXIT_CODE
