#!/bin/bash
# Monitoring script for Phase 1 training
# Usage: bash monitor_training.sh

LOG_FILE="train_phase1_improved.log"

echo "=== Phase 1 Training Monitor ==="
echo "Log file: $LOG_FILE"
echo ""

# Check if process is running
PID=$(ps aux | grep "train_sam_align_ppo" | grep -v grep | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "⚠️  Training process NOT running!"
else
    echo "✓ Training process running (PID: $PID)"
fi

echo ""
echo "=== Recent Progress ==="
tail -n 30 "$LOG_FILE" | grep -E "Iteration:|SPS:|Validation|best|dice_score" | tail -n 15

echo ""
echo "=== Training Statistics ==="
if [ -f "$LOG_FILE" ]; then
    LAST_ITER=$(grep "Iteration:" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
    LAST_SPS=$(grep "SPS:" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
    VAL_COUNT=$(grep -c "Validation at iteration" "$LOG_FILE")
    BEST_DICE=$(grep "New best validation Dice" "$LOG_FILE" | tail -n 1 | awk '{print $5}')
    
    echo "Last Iteration: $LAST_ITER"
    echo "Speed: $LAST_SPS steps/sec"
    echo "Validation checks: $VAL_COUNT"
    if [ ! -z "$BEST_DICE" ]; then
        echo "Best Validation Dice: $BEST_DICE"
    fi
fi

echo ""
echo "=== Return Distribution (last 50 episodes) ==="
grep "episodic_return" "$LOG_FILE" | tail -n 50 | awk -F'=' '{print $2}' | sed 's/\[//g' | sed 's/\]//g' | awk '{
    if ($1 == 0) zero++;
    else if ($1 > 0) pos++;
    else neg++;
    total++;
}
END {
    if (total > 0) {
        printf "Zero: %d (%.1f%%)\n", zero, (zero/total*100);
        printf "Positive: %d (%.1f%%)\n", pos, (pos/total*100);
        printf "Negative: %d (%.1f%%)\n", neg, (neg/total*100);
    }
}'

echo ""
echo "=== Recent Dice Scores (last 10) ==="
grep "dice_score" "$LOG_FILE" | tail -n 10 | awk '{print $NF}' | awk '{printf "%.4f ", $1} END {print ""}'

echo ""
echo "=== Next Validation ==="
LAST_ITER=$(grep "Iteration:" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
if [ ! -z "$LAST_ITER" ]; then
    NEXT_VAL=$((($LAST_ITER / 50 + 1) * 50))
    REMAINING=$((NEXT_VAL - $LAST_ITER))
    echo "Current: Iteration $LAST_ITER"
    echo "Next validation: Iteration $NEXT_VAL (in $REMAINING iterations)"
fi

echo ""
echo "Last updated: $(date)"
echo "To refresh, run: bash monitor_training.sh"
