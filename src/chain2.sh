#!/bin/bash
cd "$(dirname "$0")"
python3 stage2_sur.py > stage2.log 2>&1
python3 stage4_val.py > stage4.log 2>&1
python3 stage6_admissibility.py > stage6.log 2>&1
echo CHAINDONE >> stage6.log
