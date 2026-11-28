#!/bin/bash
# Sourced by scripts that must work both from Adi's Mac and on Bouchet itself (login node, compute node,
# or inside the YCRC claude container). Sets ON_BOUCHET=1 on Bouchet, 0 elsewhere, and BOUCHET_PROJ.
BOUCHET_PROJ=/nfs/roberts/project/pi_dr877/ak3288
if [ -d "$BOUCHET_PROJ/apollo-lab" ] || hostname -f 2>/dev/null | grep -q ycrc; then
  ON_BOUCHET=1
else
  ON_BOUCHET=0
fi
