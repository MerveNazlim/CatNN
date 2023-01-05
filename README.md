# CatNN
Signal categorisation NN
# Steps
This project consists of three steps. Firstly convert the root files to .h5 format. Secondly, merge all of them in one .h5 and finally training with notebook. 

## 1- Converting the ntupels to .h5
In an empty directory, follow the instructions given in a) for the first time, or b) for the other times to initilise. 
```
a)
mkdir FancyDirectory
cd FancyDirectory
mkdir build source
cp -r <pathtothepackage>/HDF5Utils_hacked/HDF5Utils source/
cp <pathtothepackage>/HDF5Utils_hacked/setup.sh source/
cd source
setupATLAS
asetup 21.2.156,AthAnalysis,here
cd ../build
cmake ../source
cmake --build ./
source */setup.sh
```
```
b)
cd FancyDirectory/source/
source setup.sh
```
There is an example script on how to run the tool, with selections etc. Please check `vll3l.sh` script.

## 2- Preprocessing 
This step will merge the files from first step and choose only the given branches. An example how the branch list is given is `preprocess/feature_lists/feature_lists_3Lvll.txt` and the file list format here `preprocess/filelists/filelists_vllsignal_3l.txt` (class number | name | path). The script can be run as follows:
```
source cvmfs-setup.sh
python preprocess_weight.py -i ./filelists/filelists_vllsignal_3l.txt --outdir ./vll -f ./feature_lists/feature_lists_3Lvll.txt -d nominal --training-size 251241
```
## 3- Training
Finally the training will use the file from step 2. People who are at IFAE, we have a google drive storage, go `https://drive.google.com` and login with you IFAE account. Copy the file from 2nd step to a folder inside your drive. Later, you will mount your drive into Colab. You should run `signalNN_vll_clean.ipynb` in the Colab, the short explanations are given inside. At the beginning, with pip install, we download certain versions of the packages. Once you run them, you will need to `Restart runtime`, and skip the downloading.
