#!/bin/bash
# Lancia le pipeline PERT per 40 e 25 nodi in parallelo.

set -e

cd /home/atorre/UTSP/unione/git/UTSP/PERT

echo "=== Lancio pipeline 40 nodi ==="
TESI_N_NODES=40 bash run_tutto.sh

echo ""
echo "=== Lancio pipeline 25 nodi ==="
TESI_N_NODES=25 bash run_tutto.sh

echo ""
echo "Tutto sottomesso. Controlla con: squeue -u $USER"
