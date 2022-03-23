import os, sys, pickle, math, argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import model_from_json
from sklearn.metrics import roc_auc_score,roc_curve, auc

seed=400
np.random.seed(seed)

def exponential_decay_fn(epoch):
  return 0.05 * 0.1**(epoch / 20)

def lr_step_decay(epoch) : #initial_lr, drop, epoch_to_drop) :

    initial_lr = 0.01
    drop = 0.9
    epoch_to_drop = 3

    if epoch >= 50 :
        epoch == 50
    elif epoch >= 25 :
        epoch -= 25
#        new_lr = initial_lr
        print('INFO Setting learning rate back to initial LR (={})'.format(initial_lr))

    new_lr = np.round(initial_lr * math.pow(drop, math.floor((1+epoch)/epoch_to_drop)),4)
    print('INFO LR Schedule: {}'.format(new_lr))
    return new_lr

def Plot_Metrics(history, path_tosave):
    for x in range(0,len(history)):
        plt.plot(history[x].history['loss'], label='Train. fold-'+str(x))
        plt.plot(history[x].history['val_loss'], label='Val. fold-'+str(x))
        #plt.title('DNN ($D_{in}$=7, $D_{hidden}$=2, 0.2Drop)')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='upper right')

    saveit = "{}/{}".format(path_tosave, "dnn_lossepo.png")
    plt.savefig(saveit)
    plt.show()

    for x in range(0,len(history)):
        plt.plot(history[x].history['accuracy'], label='Train. fold-'+str(x))
        plt.plot(history[x].history['val_accuracy'], label='Val. fold-'+str(x))
        #plt.title('DNN ($D_{in}$=7, $D_{hidden}$=2, 0.2Drop)')
        plt.xlabel('epoch')
        plt.ylabel('accuracy in %')
        plt.legend(loc='lower right')

    saveit = "{}/{}".format(path_tosave, "dnn_accepo.png")
    plt.savefig(saveit)
    plt.show()

def Save_Model(model, file_name, output_dir):
    job_suff = "_{}".format(file_name)
    arch_name = "architecture{}.json".format(job_suff)
    weights_name = "weights{}.h5".format(job_suff)

    mkdir_p(output_dir)
    arch_name = "{}/{}".format(output_dir, arch_name)
    weights_name = "{}/{}".format(output_dir, weights_name)

    print("Saving architecture to: {}".format(os.path.abspath(arch_name)))
    print("Saving weights to     : {}".format(os.path.abspath(weights_name)))
    with open(arch_name, 'w') as arch_file :
        arch_file.write(model.to_json())
    model.save_weights(weights_name)


def Load_Model(model_name, folder):
    arch_path = folder + "/architecture_{}.json".format(model_name)
    weights_path = folder + "/weights_{}.h5".format(model_name)
    print("Loading model architecture and weights ({}, {})".format(os.path.abspath(arch_path), os.path.abspath(weights_path)))
    json_file = open(os.path.abspath(arch_path), 'r')
    loaded_model = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model)
    loaded_model.load_weights(os.path.abspath(weights_path))
    return loaded_model

def Plot_NN_Output(model, train, test, log=True):
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

def plot_roc_curve(model,data,path_tosave):
    pred = model.predict(data[0])
    truth = data[1]
    fpr, tpr, thr = roc_curve(truth, pred)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             label='ROC curve (area = {:.2f})'.format(roc_auc))
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
