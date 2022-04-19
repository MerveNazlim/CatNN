import ROOT,uproot
import sys, gc, os, argparse, pickle, math, joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import tensorflow.keras.models as mod
seed = 400
np.random.seed(400)
tf.random.set_seed(400)
tf.config.run_functions_eagerly(False)
#import uproot4 as uproot
ROOT.gInterpreter.GenerateDictionary('std::vector<std::vector<int>>')
ROOT.gROOT.ProcessLine("gErrorIgnoreLevel = 2001;")

def load_model(train_path) :
    arch_path = train_path + "/architecture.json"
    weights_path = train_path + "/weights.h5"
    print(arch_path)
    print(weights_path)
    json_file = open(os.path.abspath(arch_path), 'r')
    loaded_model = json_file.read()
    json_file.close()
    loaded_model = mod.model_from_json(loaded_model)
    loaded_model.load_weights(os.path.abspath(weights_path))
    return loaded_model

def predict_NN(model, arr_frame):
    scores = np.empty([len(arr_frame), 1], dtype=np.float32)
    BATCH_INDICES = np.arange(start=0, stop=len(arr_frame), step=100000)  # row indices of batches
    BATCH_INDICES = np.append(BATCH_INDICES, len(arr_frame))  # add final batch_end row

    for index in np.arange(len(BATCH_INDICES) - 1):
        batch_start = BATCH_INDICES[index]  # first row of the batch
        batch_end = BATCH_INDICES[index + 1]  # last row of the batch
        scores[batch_start:batch_end] = model.predict(arr_frame[batch_start:batch_end])
        tf.keras.backend.clear_session()
        _ = gc.collect()
    return scores

def main() :
    parser = argparse.ArgumentParser(description = "Evaluate NN")
    parser.add_argument("-i", "--input",help = "Provide input root files", required = True)
    parser.add_argument("-t2l", "--trainpath2l", help = "path of the trainings", required = True)
    parser.add_argument("--outdir", help = "Provide an output directory for plots", default = "./")
    args = parser.parse_args()

    model2l = load_model(args.trainpath2l)

    print("Create dataframes")
    df = {}
    f1 = uproot.open(args.input,library="pd")
    f=f1['nominal']
    arr_frame2lCAT = f.arrays([
    "HT_jets",
    "HT_lep",
    "MtLepMet",
    "jet_pt0_nofwd",
    "met_met",
    "nJets_OR",
    "sumPsbtag",
    ],library="pd")

    scaler = pickle.load(open(args.trainpath2l + 'scaler.pkl','rb'))
    df_sc2l = scaler.transform(arr_frame2lCAT)

    scores2l = predict_NN(model2l ,df_sc2l)
    print(np.mean(scores2l))

    length = len(args.input.split("/"))
    print(len(args.input.split("/")))
    files = args.input.split("/")[length-1]
    file_name = args.outdir + args.input.split("/")[length-1] #11,10
    files=files.replace('./','')
    files=files.replace('.root','')
    tree_all=f.arrays(library="pd")
    tree_all=pd.concat([tree_all, pd.DataFrame({'NN_2lsstt' :np.squeeze(scores2l, axis=1)})], axis=1)
    with uproot.recreate(args.outdir+"/%s.root"%(files)) as rootfile:
       rootfile['nominal']=tree_all

    print("done, stuff saved with: {}".format(file_name))
if __name__ == "__main__" :
    main()
