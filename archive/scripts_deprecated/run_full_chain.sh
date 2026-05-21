#!/bin/bash
cd /data2/zcwang/aldolrxnmaster
CONDA_RUN="conda run -n aldol-rxn --no-capture-output"
LOG=data/v3/full_chain.log

echo "[$(date)] === FULL CHAIN START ===" > $LOG

echo "[$(date)] Step 1/3: V3 pipeline..." >> $LOG
$CONDA_RUN python scripts/run_rebuild_v3.py --date-suffix 20260515 >> $LOG 2>&1
RC=$?
echo "[$(date)] V3 exit: $RC" >> $LOG
if [ $RC -ne 0 ]; then echo "[$(date)] V3 FAILED" >> $LOG; exit 1; fi

echo "[$(date)] Step 2/3: Z/E conformers..." >> $LOG
$CONDA_RUN python scripts/run_ze_conformers.py >> $LOG 2>&1
RC=$?
echo "[$(date)] Z/E exit: $RC" >> $LOG
if [ $RC -ne 0 ]; then echo "[$(date)] Z/E FAILED" >> $LOG; exit 1; fi

echo "[$(date)] Step 3/3: MechAware model..." >> $LOG
$CONDA_RUN python scripts/run_mechaware_model.py >> $LOG 2>&1
RC=$?
echo "[$(date)] MechAware exit: $RC" >> $LOG

echo "[$(date)] === ALL DONE ===" >> $LOG
