#!/bin/bash
#SBATCH --job-name=tesi_B
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=output_B_%j.txt
#SBATCH --error=error_B_%j.txt
module load python
module load gurobi
cd /home/atorre/UTSP/unione/git
source venv/bin/activate
python main.py --only B
