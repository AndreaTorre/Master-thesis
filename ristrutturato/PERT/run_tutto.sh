#!/bin/bash
# Lancia tutta la pipeline PERT con dipendenze SLURM automatiche.
# Uso base:
#   bash run_tutto.sh
#
# Uso con istanza specifica:
#   TESI_DATA_FILE=/home/atorre/UTSP/unione/git/UTSP/data/pert/nodi_25.json \
#   TESI_OUTPUT_DIR=/home/atorre/UTSP/unione/git/UTSP/PERT/RISULTATI_25 \
#   bash run_tutto.sh

set -e

cd /home/atorre/UTSP/unione/git/UTSP/PERT

export TESI_EXPERIMENT=PERT
export TESI_N_NODES=${1:-${TESI_N_NODES:-40}}
export TESI_OUTPUT_DIR=/home/atorre/UTSP/unione/git/UTSP/PERT/RISULTATI_${TESI_N_NODES}

export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/atorre/UTSP/unione/git/UTSP/common:$PYTHONPATH

mkdir -p "$TESI_OUTPUT_DIR/output"
mkdir -p "$TESI_OUTPUT_DIR/grafici"
mkdir -p "$TESI_OUTPUT_DIR/checkpoint"
mkdir -p "$TESI_OUTPUT_DIR/pkl"

echo "=== Pipeline parallela Esperimento B PERT ==="
echo "TESI_N_NODES=$TESI_N_NODES"
echo "TESI_OUTPUT_DIR=$TESI_OUTPUT_DIR"

SETUP=$(sbatch --parsable \
  --output="$TESI_OUTPUT_DIR/output/output_setup_%j.txt" \
  --error="$TESI_OUTPUT_DIR/output/error_setup_%j.txt" \
  --export=ALL,TESI_EXPERIMENT=PERT,TESI_N_NODES="$TESI_N_NODES" \
  run_0_setup.sh)
echo "Setup:    job $SETUP"

PI=$(sbatch --parsable \
  --dependency=afterok:$SETUP \
  --output="$TESI_OUTPUT_DIR/output/output_pi_%j.txt" \
  --error="$TESI_OUTPUT_DIR/output/error_pi_%j.txt" \
  --export=ALL,TESI_EXPERIMENT=PERT,TESI_N_NODES="$TESI_N_NODES" <<'EOF_PI'
#!/bin/bash
#SBATCH --job-name=B_pi
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

module load python
module load gurobi

cd /home/atorre/UTSP/unione/git/UTSP/PERT
source /home/atorre/UTSP/unione/git/UTSP/venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/atorre/UTSP/unione/git/UTSP/common:$PYTHONPATH

python gurobi_parallelo.py pi
EOF_PI
)
echo "PI:       job $PI (dopo $SETUP)"

EEV=$(sbatch --parsable \
  --dependency=afterok:$SETUP \
  --output="$TESI_OUTPUT_DIR/output/output_eev_%j.txt" \
  --error="$TESI_OUTPUT_DIR/output/error_eev_%j.txt" \
  --export=ALL,TESI_EXPERIMENT=PERT,TESI_N_NODES="$TESI_N_NODES" <<'EOF_EEV'
#!/bin/bash
#SBATCH --job-name=B_eev
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

module load python
module load gurobi

cd /home/atorre/UTSP/unione/git/UTSP/PERT
source /home/atorre/UTSP/unione/git/UTSP/venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/atorre/UTSP/unione/git/UTSP/common:$PYTHONPATH

python gurobi_parallelo.py eev
EOF_EEV
)
echo "EEV:      job $EEV (dopo $SETUP)"

STO=$(sbatch --parsable \
  --dependency=afterok:$SETUP \
  --output="$TESI_OUTPUT_DIR/output/output_sto_%j.txt" \
  --error="$TESI_OUTPUT_DIR/output/error_sto_%j.txt" \
  --export=ALL,TESI_EXPERIMENT=PERT,TESI_N_NODES="$TESI_N_NODES" <<'EOF_STO'
#!/bin/bash
#SBATCH --job-name=B_sto
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load python
module load gurobi

cd /home/atorre/UTSP/unione/git/UTSP/PERT
source /home/atorre/UTSP/unione/git/UTSP/venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/atorre/UTSP/unione/git/UTSP/common:$PYTHONPATH

python gurobi_parallelo.py sto
EOF_STO
)
echo "STO:      job $STO (dopo $SETUP)"

ASM=$(sbatch --parsable \
  --dependency=afterok:$PI:$EEV:$STO \
  --output="$TESI_OUTPUT_DIR/output/output_assemble_%j.txt" \
  --error="$TESI_OUTPUT_DIR/output/error_assemble_%j.txt" \
  --export=ALL,TESI_EXPERIMENT=PERT,TESI_N_NODES="$TESI_N_NODES" \
  run_2_assemble.sh)
echo "Assemble: job $ASM (dopo $PI, $EEV, $STO)"

echo ""
echo "Pipeline sottomessa. Controlla con: squeue -u $USER"
