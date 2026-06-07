#!/bin/bash
#SBATCH --job-name=grid_array
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=grid_logs/output_%a.txt
#SBATCH --error=grid_logs/error_%a.txt

module load python
module load gurobi
cd /home/atorre/UTSP/unione/git
source venv/bin/activate

python run_single.py --combo-index $SLURM_ARRAY_TASK_ID