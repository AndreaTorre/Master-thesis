#!/bin/bash
# Lancia Esperimento B per PERT/CVETT e 15/25/40 nodi.
#
# Uso:
#   bash submit_b.sh PERT 40
#   bash submit_b.sh CVETT 25
#
# oppure:
#   TESI_EXPERIMENT=PERT TESI_N_NODES=40 bash submit_b.sh

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

mkdir -p "$OUTPUT_DIR/output"
mkdir -p "$OUTPUT_DIR/grafici"
mkdir -p "$OUTPUT_DIR/checkpoint"
mkdir -p "$OUTPUT_DIR/pkl"

echo "=== Submit B ==="
echo "EXPERIMENT=$EXP"
echo "N_NODES=$N"
echo "OUTPUT_DIR=$OUTPUT_DIR"

sbatch \
  --job-name="${EXP}_${N}_B" \
  --time=24:00:00 \
  --cpus-per-task=8 \
  --mem=32G \
  --output="$OUTPUT_DIR/output/output_B_%j.txt" \
  --error="$OUTPUT_DIR/output/error_B_%j.txt" \
  --export=ALL,TESI_EXPERIMENT="$EXP",TESI_N_NODES="$N" <<'EOF_JOB'
#!/bin/bash

module load python
module load gurobi

source /home/atorre/UTSP/unione/git/UTSP/venv/bin/activate

cd /home/atorre/UTSP/unione/git/UTSP/common

export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/atorre/UTSP/unione/git/UTSP/common:$PYTHONPATH

python main.py --only B
EOF_JOB
