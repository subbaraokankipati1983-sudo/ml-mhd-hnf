#!/bin/bash
cd "$(dirname "$0")"
for t in phifloor cap10 cap15 cap20 cap25; do python3 stage3_de.py $t 1; done
echo RESTDONE
