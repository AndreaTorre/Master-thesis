#!/bin/bash
#SBATCH --job-name=grid_array
#SBATCH --array=0-71        # cambia in base al totale combinazioni
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=grid_logs/output_%a.txt
#SBATCH --error=grid_logs/error_%a.txt

module load python
module load gurobi
cd /home/atorre/UTSP/unione/git
source venv/bin/activate

python run_single.py --combo-index $SLURM_ARRAY_TASK_ID