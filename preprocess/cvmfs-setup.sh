
#!/bin/bash

setupATLAS

export PYTHONPATH=${PWD}:${PYTHONPATH}
export PYTHONPATH=${PWD}/analyze:${PYTHONPATH}
#lsetup "lcgenv -p LCG_98python3 x86_64-centos7-gcc8-opt tensorflow" > /dev/null  # TF 1.8.0 and python3
lsetup "lcgenv -p LCG_98python3 x86_64-centos7-gcc8-opt h5py" > /dev/null
lsetup "lcgenv -p LCG_98python3 x86_64-centos7-gcc8-opt scikitlearn" > /dev/null
lsetup "lcgenv -p LCG_98python3 x86_64-centos7-gcc8-opt keras"  > /dev/null
#lsetup "lcgenv -p LCG_98python3 x86_64-centos7-gcc8-opt keras_applications" > /dev/null
#lsetup "lcgenv -p LCG_98python3 x86_64-centos7-gcc8-opt keras_preprocessing" > /dev/null
lsetup "lcgenv -p LCG_98python3 x86_64-centos7-gcc8-opt matplotlib" > /dev/null
