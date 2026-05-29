#!/bin/bash
#SBATCH --job-name=tesi_B_UTSP
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
module load python
module load gurobi
cd /home/atorre/UTSP/unione/git
source venv/bin/activate
echo "Python usato:"
which python
python -c "import sys; print(sys.executable)"
echo "Controllo numpy:"
python -c "import numpy; print(numpy.__version__)"
python main.py --only B_UTSP_LS
