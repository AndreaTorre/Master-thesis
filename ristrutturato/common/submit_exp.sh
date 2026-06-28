#!/bin/bash
# Lancia Esperimento B_UTSP_LS per PERT/CVETT e 15/25/40 nodi.
#
# Uso:
#   bash submit_exp.sh PERT 40
#   bash submit_exp.sh CVETT 25
#
# oppure:
#   TESI_EXPERIMENT=PERT TESI_N_NODES=40 bash submit_exp.sh

set -e

EXP=${1:-${TESI_EXPERIMENT:-PERT}}
N=${2:-${TESI_N_NODES:-40}}

EXP=$(echo "$EXP" | tr '[:lower:]' '[:upper:]')

if [[ "$EXP" != "PERT" && "$EXP" != "CVETT" ]]; then
  echo "Errore: esperimento non valido: $EXP. Usa PERT oppure CVETT."
  exit 1
fi

if [[ "$N" != "15" && "$N" != "25" && "$N" != "40" ]]; then
  echo "Errore: numero nodi non valido: $N. Usa 15, 25 oppure 40."
  exit 1
fi

ROOT=/home/atorre/UTSP/unione/git/UTSP
COMMON_DIR=$ROOT/common
OUTPUT_DIR=$ROOT/$EXP/RISULTATI_$N
if [[ -n "$TESI_BATCH_SWEEP" ]]; then
  OUTPUT_DIR=$OUTPUT_DIR/batch_sweep/BATCH_$TESI_BATCH_SWEEP
fi

mkdir -p "$OUTPUT_DIR/output"
mkdir -p "$OUTPUT_DIR/grafici"
mkdir -p "$OUTPUT_DIR/checkpoint"
mkdir -p "$OUTPUT_DIR/pkl"

echo "=== Submit B_UTSP_LS ==="
echo "EXPERIMENT=$EXP"
echo "N_NODES=$N"
echo "OUTPUT_DIR=$OUTPUT_DIR"


DEPENDENCY_ARG=""
if [[ -n "$TESI_DEPENDENCY" ]]; then
  DEPENDENCY_ARG="--dependency=afterok:$TESI_DEPENDENCY"
fi

sbatch \
  --job-name="${EXP}_${N}_UTSP" \
  --time=04:00:00 \
  --cpus-per-task=8 \
  --mem=16G \
  --output="$OUTPUT_DIR/output/output_UTSP_%j.txt" \
  --error="$OUTPUT_DIR/output/error_UTSP_%j.txt" \
  --export=ALL,TESI_EXPERIMENT="$EXP",TESI_N_NODES="$N" \
  $DEPENDENCY_ARG <<'EOF_JOB'
#!/bin/bash

module load python
module load gurobi

source /home/atorre/UTSP/unione/git/UTSP/venv/bin/activate

cd /home/atorre/UTSP/unione/git/UTSP/common

export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/atorre/UTSP/unione/git/UTSP/common:$PYTHONPATH

echo "Python usato:"
which python
python -c "import sys; print(sys.executable)"

echo "Controllo numpy:"
python -c "import numpy; print(numpy.__version__)"

python main.py --only B_UTSP_LS
EOF_JOB
