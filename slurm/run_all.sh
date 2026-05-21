#!/bin/bash
#SBATCH --job-name=tesi_B_UTSP
#SBATCH --output=slurm-%j.out
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

module load python || true
source .venv/bin/activate

python main.py --only ALL
