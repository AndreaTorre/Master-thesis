#!/bin/bash

# Sottomette i batch uno alla volta, aspettando che il precedente finisca.
# Uso: bash slurm/submit_all.sh

mkdir -p grid_logs
# per grid su mean
#BATCHES=(
#    "0-971"
#    "972-1943"
#    "1944-2591"
#)


# per grid su sum
BATCHES=(
    "0-971"
    "972-1943"
    "1944-2915"
    "2916-3887"
)


for i in "${!BATCHES[@]}"; do
    RANGE="${BATCHES[$i]}"

    JID=$(sbatch --array=$RANGE --parsable slurm/run_all.sh)

    echo "$(date) — Batch $((i+1))/${#BATCHES[@]} sottomesso: job $JID (combo $RANGE)"
    echo "Aspetto che il job $JID finisca..."

    while squeue -u $USER -j $JID 2>/dev/null | grep -q $JID; do
        REMAINING=$(squeue -u $USER -j $JID 2>/dev/null | grep -c $JID)
        echo "  $(date +%H:%M) — job $JID: $REMAINING task ancora in coda/running"
        sleep 120
    done

    echo "$(date) — Batch $((i+1)) completato."
done

echo ""
echo "Tutti i batch completati. Lancia: python collect_results.py"