#!/bin/sh

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh

asetup 21.2.156,AthAnalysis,here
cd ../build
source */setup.sh
cd ../source
