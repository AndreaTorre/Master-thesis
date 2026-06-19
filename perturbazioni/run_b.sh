#!/bin/bash
#SBATCH --job-name=tesi_B
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=output_B_%j.txt
#SBATCH --error=error_B_%j.txt

module load python
module load gurobi
cd /home/atorre/UTSP/unione/git/pert/
source /home/atorre/UTSP/unione/git/venv/bin/activate

export PYTHONUNBUFFERED=1
python main.py --only B
