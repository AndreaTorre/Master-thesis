#!/bin/bash
#SBATCH --job-name=B_setup
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/atorre/UTSP/unione/git/UTSP/PERT/RISULTATI_40/output/output_setup_%j.txt
#SBATCH --error=/home/atorre/UTSP/unione/git/UTSP/PERT/RISULTATI_40/output/error_setup_%j.txt

module load python
module load gurobi

cd /home/atorre/UTSP/unione/git/UTSP/PERT
source /home/atorre/UTSP/unione/git/UTSP/venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/atorre/UTSP/unione/git/UTSP/common:$PYTHONPATH

export TESI_EXPERIMENT=PERT
export TESI_N_NODES=${TESI_N_NODES:-40}

python gurobi_parallelo.py setup
