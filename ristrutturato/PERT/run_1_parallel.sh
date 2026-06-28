#!/bin/bash
set -e

N=${1:-${TESI_N_NODES:-40}}

if [[ "$N" != "15" && "$N" != "25" && "$N" != "40" ]]; then
  echo "Errore: N deve essere 15, 25 oppure 40. Valore ricevuto: $N"
  exit 1
fi

ROOT=/home/atorre/UTSP/unione/git/UTSP
PERT_DIR=$ROOT/PERT
OUTPUT_DIR=$PERT_DIR/RISULTATI_$N

mkdir -p "$OUTPUT_DIR/output" "$OUTPUT_DIR/grafici" "$OUTPUT_DIR/checkpoint" "$OUTPUT_DIR/pkl"

DEP=""
if [[ -n "${SETUP_ID:-}" ]]; then
  DEP="--dependency=afterok:$SETUP_ID"
fi

submit_phase () {
  PHASE=$1
  TIME=$2
  CPU=$3
  MEM=$4

  sbatch --parsable $DEP \
    --job-name="B_${PHASE}_${N}" \
    --time="$TIME" \
    --cpus-per-task="$CPU" \
    --mem="$MEM" \
    --output="$OUTPUT_DIR/output/output_${PHASE}_%j.txt" \
    --error="$OUTPUT_DIR/output/error_${PHASE}_%j.txt" \
    --export=ALL,TESI_EXPERIMENT=PERT,TESI_N_NODES="$N" \
    --wrap "module load python; module load gurobi; source $ROOT/venv/bin/activate; cd $PERT_DIR; export PYTHONUNBUFFERED=1; export PYTHONPATH=$ROOT/common:\$PYTHONPATH; python gurobi_parallelo.py $PHASE"
}

echo "=== Lancio PI/EEV/STO per PERT, N=$N ==="

PI=$(submit_phase pi 06:00:00 4 8G)
EEV=$(submit_phase eev 06:00:00 4 8G)
STO=$(submit_phase sto 24:00:00 8 32G)

echo "PI:  $PI"
echo "EEV: $EEV"
echo "STO: $STO"
echo ""
echo "Per assemblare dopo:"
echo "sbatch --dependency=afterok:$PI:$EEV:$STO --export=ALL,TESI_EXPERIMENT=PERT,TESI_N_NODES=$N run_2_assemble.sh"
