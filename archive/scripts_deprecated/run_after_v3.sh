#!/bin/bash
PYTHON_PID=3700129
LOG_DIR=/data2/zcwang/aldolrxnmaster/data/v3/mechaware
CONDA_RUN="conda run -n aldol-rxn --no-capture-output"
cd /data2/zcwang/aldolrxnmaster

echo "[$(date)] Waiting for V3 Python (PID $PYTHON_PID)..."
while kill -0 $PYTHON_PID 2>/dev/null; do sleep 30; done
echo "[$(date)] V3 done."

echo "[$(date)] Step 2: Z/E conformers..."
$CONDA_RUN python scripts/run_ze_conformers.py > $LOG_DIR/ze_conformers_stdout.log 2>&1
echo "[$(date)] Z/E exit: $?"

echo "[$(date)] Step 3: MechAware model..."
$CONDA_RUN python scripts/run_mechaware_model.py > $LOG_DIR/mechaware_model_stdout.log 2>&1
echo "[$(date)] MechAware exit: $?"

echo "[$(date)] ALL DONE! Check: cat $LOG_DIR/mechaware_results.json"
