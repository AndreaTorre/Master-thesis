#!/bin/bash
# Lancia lo sweep su UTSP_BATCH_SIZE per un esperimento/dimensione dati.
#
# Uso:
#   bash run_batch_sweep.sh PERT 25
#   bash run_batch_sweep.sh PERT 25 "20 30 40 50"   # batch size custom

set -e
EXP=${1:?Specifica EXP (PERT o CVETT)}
N=${2:?Specifica N (15, 25 o 40)}
SIZES=${3:-"20 30 40 50"}

for B in $SIZES; do
  echo "=== Sweep ${EXP} ${N} nodi — batch size ${B} ==="
  TESI_BATCH_SWEEP=$B bash submit_exp.sh "$EXP" "$N"
done
