import os, sys, pickle, math, argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import model_from_json
from sklearn.metrics import roc_auc_score,roc_curve, auc,accuracy_score

seed=400
np.random.seed(seed)

def exponential_decay_fn(epoch):
  return 0.05 * 0.1**(epoch / 20)

def Find_Best_Fold(history_list):
    fold_best_val_acc = []
    for hist in history_list:
        fold_best_val_acc.append(np.max(hist.history['val_accuracy']))
    best_fold_idx = np.argmax(fold_best_val_acc)
    print('Best fold is fold number', best_fold_idx)
    return best_fold_idx

def lr_step_decay(epoch) : #initial_lr, drop, epoch_to_drop) :

    initial_lr = 0.01
    drop = 0.9
    epoch_to_drop = 3

    if epoch >= 50 :
        epoch == 50
    elif epoch >= 25 :
        epoch -= 25
#        new_lr = initial_lr
        #print('INFO Setting learning rate back to initial LR (={})'.format(initial_lr))

    new_lr = np.round(initial_lr * math.pow(drop, math.floor((1+epoch)/epoch_to_drop)),4)
    #print('INFO LR Schedule: {}'.format(new_lr))
    return new_lr

def Plot_Metrics_KFold(history, path_tosave):
    mkdir_p(path_tosave)
    for x in range(0,len(history)):
        plt.plot(history[x].history['loss'], label='Train. fold-'+str(x))
        plt.plot(history[x].history['val_loss'], label='Val. fold-'+str(x))
        #plt.title('DNN ($D_{in}$=7, $D_{hidden}$=2, 0.2Drop)')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='upper right')

    saveit = "{}/{}".format(path_tosave, "Loss_KFold.png")
    plt.savefig(saveit)
    plt.show()

    for x in range(0,len(history)):
        plt.plot(history[x].history['accuracy'], label='Train. fold-'+str(x))
        plt.plot(history[x].history['val_accuracy'], label='Val. fold-'+str(x))
        #plt.title('DNN ($D_{in}$=7, $D_{hidden}$=2, 0.2Drop)')
        plt.xlabel('epoch')
        plt.ylabel('accuracy in %')
        plt.legend(loc='lower right')

    saveit = "{}/{}".format(path_tosave, "Acc_KFold.png")
    plt.savefig(saveit)
    plt.show()

def Plot_Metrics(history, path_tosave):
    mkdir_p(path_tosave)
    for x in range(0,len(history)):
        plt.plot(history[x].history['loss'], label='Train Data')
        plt.plot(history[x].history['val_loss'], label='Validation Data')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='upper right')

    saveit = "{}/{}".format(path_tosave, "Loss.png")
    plt.savefig(saveit)
    plt.show()

    for x in range(0,len(history)):
        plt.plot(history[x].history['accuracy'], label='Train Data')
        plt.plot(history[x].history['val_accuracy'], label='Validation Data')
        plt.xlabel('epoch')
        plt.ylabel('accuracy in %')
        plt.legend(loc='lower right')

    saveit = "{}/{}".format(path_tosave, "Acc.png")
    plt.savefig(saveit)
    plt.show()

def Save_Model(model, output_dir):
    arch_name = "architecture.json"
    weights_name = "weights.h5"

    mkdir_p(output_dir)
    arch_name = "{}/{}".format(output_dir, arch_name)
    weights_name = "{}/{}".format(output_dir, weights_name)

    print("Saving architecture to: {}".format(os.path.abspath(arch_name)))
    print("Saving weights to     : {}".format(os.path.abspath(weights_name)))
    with open(arch_name, 'w') as arch_file :
        arch_file.write(model.to_json())
    model.save_weights(weights_name)


def Load_Model(output_dir):
    arch_path = output_dir + "/architecture.json"
    weights_path = output_dir + "/weights.h5"
    print("Loading model architecture and weights ({}, {})".format(os.path.abspath(arch_path), os.path.abspath(weights_path)))
    json_file = open(os.path.abspath(arch_path), 'r')
    loaded_model = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model)
    loaded_model.load_weights(os.path.abspath(weights_path))
    return loaded_model

def Plot_NN_Output(model, train, test, path_tosave, log=True):
    nn_scores_test = model.predict(test[0], verbose = True)
    nn_scores = model.predict(train[0], verbose = True)
    fig = plt.figure()
    plt.grid(color='k', which='both', linestyle='--', lw=0.5, alpha=0.1, zorder = 0)
    plt.xlabel("NN output", horizontalalignment='right', x=1)
    plt.xlim([0,1])
    plt.ylabel("Density")
    if log == True:
        plt.yscale('log')
    histargs = {"bins":40, "range":(0,1.), "density":True, "histtype":'step'}
    plt.hist(nn_scores_test[test[1]==1],label = "Test_Signal", **histargs)
    plt.hist(nn_scores_test[test[1]==0],label = "Test_Background", **histargs)
    plt.hist(nn_scores[train[1]==1],label = "Train_Signal", **histargs)
    plt.hist(nn_scores[train[1]==0],label = "Train_Background", **histargs)
    plt.legend(loc='upper center', frameon=False,)
    saveit = "{}/{}".format(path_tosave, "DNN_Output.png")
    plt.savefig(saveit)

def plot_roc_curve(model,data,path_tosave):
    pred = model.predict(data[0])
    truth = data[1]
    fpr, tpr, thr = roc_curve(truth, pred)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             label='ROC curve (area = {:.4f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(path_tosave+"/ROC.png")
    plt.show()

def mkdir_p(path) :
    import errno
    """
    Make a directory, if it exists silence the exception
    Args:
        path : full directory path to be made
    """
    try :
        os.makedirs(path)
    except OSError as exc :
        if exc.errno == errno.EEXIST and os.path.isdir(path) :
            pass
        else :
            raise

def Find_Eff_Cut(Sig, Bkg, Signal_Eff):
    Cut = np.linspace(0,1,10000)
    i = 0
    Sig_Eff = Sig[Sig > Cut[i]].size/Sig.size
    while Sig_Eff > Signal_Eff:
        if Cut[i] == Cut[-1]:
            print('Cant find desiered Signal Efficency!')
            return 0, 0
        i += 1
        Sig_Eff = Sig[Sig > Cut[i]].size/Sig.size
    Bkg_Eff = Bkg[Bkg > Cut[i]].size/Bkg.size
    print(f"Cut_Value: {Cut[i]:.3f}")
    print(f"Sig_Eff: {Sig_Eff:.3f}")
    print(f"Bkg_Eff: {Bkg_Eff:.3f}")
    print(f"Bkg Rejection: {1/Bkg_Eff:.3f}")
    return Sig_Eff, Bkg_Eff, Cut[i]

def get_feature_importance(test, model, Signal_Cut, n):
    f = []
    g = []
    y_pred = model.predict(test[0])
    y_pred[y_pred > Signal_Cut] = 1
    y_pred[y_pred <= Signal_Cut] = 0
    s = accuracy_score(test[1], y_pred)
    for j in range(test[0].shape[1]):
        total = []
        for i in range(n):
            perm = np.random.permutation(range(test[0].shape[0]))
            X_test_ = test[0].copy()
            X_test_[:, j] = test[0][perm, j]
            y_pred_ = model.predict(X_test_)
            y_pred_[y_pred_ > Signal_Cut] = 1
            y_pred_[y_pred_ <= Signal_Cut] = 0
            s_ij = accuracy_score(test[1], y_pred_)
            total.append(s_ij)
        total = np.array(total)
        f.append(s - total.mean())
        g.append(total.std())
    return f, g

def Make_Confusion_Matrix(test, Predicted, Signal_Cut, class_names, path_tosave, relativ=True):
    NN_Cutted = np.array(Predicted)
    NN_Cutted[Predicted > Signal_Cut] = 1
    NN_Cutted[Predicted < Signal_Cut] = 0
    No_Classes = test[3].max()+1
    C_M = np.zeros([No_Classes, 2])
    fig, ax = plt.subplots(figsize=(10,10))
    if relativ == True:
        for i in np.arange(No_Classes):
            C_M[i,0] = NN_Cutted[test[3]==i].mean()
            C_M[i,1] = abs(1-C_M[i,0])
        ax = sns.heatmap(C_M, annot=True, fmt='.2%', cmap='Blues')
    else:
        for i in np.arange(No_Classes):
            C_M[i,0] = np.sum([NN_Cutted[test[3]==i]==1])
            C_M[i,1] = np.array([NN_Cutted[test[3]==i]==1]).size - C_M[i,0]
        ax = sns.heatmap(C_M, annot=True, fmt=".0f", cmap='Blues')

    ax.set_xlabel('\nPredicted Class')
    ax.set_ylabel('Class ');

    ax.xaxis.set_ticklabels(['Signal','Background'])
    ax.yaxis.set_ticklabels(class_names)

    if relativ==True:
        save_path = path_tosave + "/Confusion_Matrix_Rel.png"
    else:
        save_path = path_tosave + "/Confusion_Matrix_Abs.png"
    plt.savefig(save_path)
    plt.show()

def ScaleWeights(y,w):
    sum_wpos = sum( w[i] for i in range(len(y)) if y[i] == 1.0  )
    sum_wneg = sum( w[i] for i in range(len(y)) if y[i] == 0.0  )

    for i in range(len(w)):
        if (y[i]==1.0):
            w[i] = w[i] * (0.5/sum_wpos)
        else:
            w[i] = w[i] * (0.5/sum_wneg)

    w_av = sum(w)/len(w)
    w[:] = [x/w_av for x in w]

    sum_wpos_check = sum( w[i] for i in range(len(y)) if y[i] == 1.0  )
    sum_wneg_check = sum( w[i] for i in range(len(y)) if y[i] == 0.0  )

    print ('\n======Weight Statistic========================================')
    print ('Weights::        W(1)=%g, W(0)=%g' % (sum_wpos, sum_wneg))
    print ('Scaled weights:: W(1)=%g, W(0)=%g' % (sum_wpos_check, sum_wneg_check))
    print ('==============================================================')
