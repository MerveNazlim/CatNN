from util import *

seed=400

import tensorflow as tf
tf.random.set_seed(seed)
from tensorflow import keras
# keras
from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import models as Km
from tensorflow.keras import layers as Kl
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import tensorflow.keras
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score,roc_curve, auc
from sklearn.preprocessing import scale, normalize
from sklearn.preprocessing import RobustScaler, StandardScaler,minmax_scale
from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold
from sklearn.model_selection import train_test_split

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

def Train_Val_Test_Split(input, targets, weights):

    input, X_val, weights, weights_val, targets, y_val = train_test_split(input, weights, targets, test_size=0.1, random_state=42)
    X_train, X_test, weights_train, weights_test, y_train, y_test = train_test_split(input, weights, targets, test_size=0.2, random_state=42)

    return (X_train, y_train, weights_train), (X_val, y_val, weights_val), (X_test, y_test, weights_test)

def Create_Model_basic(input_shape):
    layer_opts = dict( activation = 'sigmoid', kernel_initializer = initializers.glorot_normal(seed=seed))
    input_layer = Kl.Input(shape = input_shape )
    x = Kl.Dense( 36, **layer_opts) (input_layer)
    x = Kl.Dropout(0.4)(x)
    x = Kl.Dense( 48, **layer_opts) (x)
    y_pred = Kl.Dense( 1., activation = 'sigmoid', name = "OutputLayer" )(x)
    model = Km.Model(inputs= input_layer, outputs=y_pred )
    model_optimizer = Adam(lr=0.0001)
    model.compile(optimizer=tf.keras.optimizers.Adam(),loss='binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model

def Train_NN(model, train, val, n_epochs = 400, batch_size = 2000,):
    fit_history_list = []
    #lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_step_decay)

    fit_history_list.append(model.fit(train[0], train[1], epochs = n_epochs, shuffle = True, batch_size = batch_size, validation_data = (val[0],val[1]),
    sample_weight = train[2], callbacks=[tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20, verbose = True, min_delta = 0.001) ,lr_schedule])) #

    return fit_history_list

def Train_NN_old(input_scaled, targets, weights, n_epochs = 400, batch_size = 2000, num_folds = 2):
    auc_scores = []
    auc_targets = []
    acc_per_fold = []
    loss_per_fold = []
    results = []
    model_history = []
    fitfull_history = []
    callbacks = []
    kfold = KFold(num_folds, True, 1)
    fold_no = 1
    for train, test in kfold.split(input_scaled, targets):
        train_data = input_scaled[train]
        test_data = input_scaled[test]
        train_label = targets[train]
        test_label = targets[test]

        layer_opts = dict( activation = 'sigmoid', kernel_initializer = initializers.glorot_normal(seed=seed))
        input_layer = Kl.Input(shape = (input_scaled.shape[1],) )
        x = Kl.Dense( 36, **layer_opts) (input_layer)
        x = Kl.Dropout(0.4)(x)
        x = Kl.Dense( 48, **layer_opts) (x)
        y_pred = Kl.Dense( 1., activation = 'sigmoid', name = "OutputLayer" )(x)
        model = Km.Model(inputs= input_layer, outputs=y_pred )
        model_optimizer = Adam(lr=0.0001)
        model.compile(optimizer=tf.keras.optimizers.Adam(),loss='binary_crossentropy', metrics = ['accuracy'])
        model.summary()
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_step_decay)
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        # Fit data to model
        w_train = weights[train]
        #train_0, train_1 = len(train_label[train_label==0]), len(train_label[train_label==1])
        #test_0, test_1 = len(test_label[test_label==0]), len(test_label[test_label==1])
        #print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
        fit_history = model.fit(train_data, train_label, epochs = n_epochs, shuffle = True, batch_size = batch_size ,validation_data = (test_data,test_label), sample_weight=w_train,callbacks=[tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20, verbose = True, min_delta = 0.001),lr_schedule]) #
        scores = model.evaluate(test_data, test_label,batch_size=batch_size, verbose=0)
        nn_scores = model.predict(test_data,verbose = True)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        results.append(scores)
        model_history.append(model)
        fitfull_history.append(fit_history)
        auc_scores.append(nn_scores)
        auc_targets.append(targets[test])
        fold_no += 1

    return acc_per_fold, loss_per_fold, results, model_history, fitfull_history, auc_scores, auc_targets
