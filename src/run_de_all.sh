#!/bin/bash
cd "$(dirname "$0")"
for s in 1 2 3 4 5 6 7 8 9 10; do python3 stage3_de.py ref $s; done
for t in w2cf whcf w2sh whsh D0 shear nubulk wide phifloor cap10 cap15 cap20 cap25; do python3 stage3_de.py $t 1; done
echo ALLDONE
