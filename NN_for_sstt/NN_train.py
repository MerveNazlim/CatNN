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
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import roc_auc_score,roc_curve, auc
from sklearn.preprocessing import scale, normalize
from sklearn.preprocessing import RobustScaler, StandardScaler,minmax_scale
from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold
from sklearn.model_selection import train_test_split

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

def distance_corr(var_1, var_2, normedweight, power=1):
    """var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation

    va1_1, var_2 and normedweight should all be 1D tf tensors with the same number of entries

    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """

    xx = tf.reshape(var_1, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_1)])
    xx = tf.reshape(xx, [tf.size(var_1), tf.size(var_1)])

    yy = tf.transpose(xx)
    amat = tf.math.abs(xx-yy)

    xx = tf.reshape(var_2, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_2)])
    xx = tf.reshape(xx, [tf.size(var_2), tf.size(var_2)])

    yy = tf.transpose(xx)
    bmat = tf.math.abs(xx-yy)

    amatavg = tf.reduce_mean(amat*normedweight, axis=1)
    bmatavg = tf.reduce_mean(bmat*normedweight, axis=1)

    minuend_1 = tf.tile(amatavg, [tf.size(var_1)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_1), tf.size(var_1)])
    minuend_2 = tf.transpose(minuend_1)
    Amat = amat-minuend_1-minuend_2+tf.reduce_mean(amatavg*normedweight)

    minuend_1 = tf.tile(bmatavg, [tf.size(var_2)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_2), tf.size(var_2)])
    minuend_2 = tf.transpose(minuend_1)
    Bmat = bmat-minuend_1-minuend_2+tf.reduce_mean(bmatavg*normedweight)

    ABavg = tf.reduce_mean(Amat*Bmat*normedweight,axis=1)
    AAavg = tf.reduce_mean(Amat*Amat*normedweight,axis=1)
    BBavg = tf.reduce_mean(Bmat*Bmat*normedweight,axis=1)

    if power==1:
        dCorr = tf.reduce_mean(ABavg*normedweight)/tf.math.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight))
    elif power==2:
        dCorr = (tf.reduce_mean(ABavg*normedweight))**2/(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight))
    else:
        dCorr = (tf.reduce_mean(ABavg*normedweight)/tf.math.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight)))**power

    return dCorr

def Train_Val_Test_Split(input, targets, weights, class_labels):

    input, X_val, weights, weights_val, targets, y_val, class_labels, class_labels_val = train_test_split(input, weights, targets, class_labels, test_size=0.1, random_state=42)
    X_train, X_test, weights_train, weights_test, y_train, y_test, class_labels_train, class_labels_test = train_test_split(input, weights, targets, class_labels, test_size=0.2, random_state=42)

    return (X_train, y_train, weights_train, class_labels_train), (X_val, y_val, weights_val, class_labels_val), (X_test, y_test, weights_test, class_labels_test)

def Create_Model_basic(input_shape):
    layer_opts = dict( activation = 'sigmoid', kernel_initializer = initializers.glorot_normal(seed=seed))
    input_layer = Kl.Input(shape = input_shape )
    x = Kl.Dense( 36, **layer_opts) (input_layer)
    x = Kl.Dense( 48, **layer_opts) (x)
    y_pred = Kl.Dense( 1., activation = 'sigmoid', name = "OutputLayer" )(x)
    model = Km.Model(inputs= input_layer, outputs=y_pred )
    model_optimizer = Adam(lr=0.0001)
    model.compile(optimizer=tf.keras.optimizers.Adam(),loss='binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model

#def Loss_with_Disco(y_true, y_pred, Dphi, weight, lamb=0.1):
#    return (distance_corr(Dphi, y_pred, weight) * lamb)

def Loss_with_Disco(y_true, y_pred, Dphi, weight, lamb=0.1):
    BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return (distance_corr(Dphi, y_pred, weight) * lamb + BCE(y_true, y_pred))

def Loss_Disco(Dphi_and_weight, lamb):
    Dphi = Dphi_and_weight[0]
    weight = Dphi_and_weight[1]
    def Disco(y_true, y_pred):
        return Loss_with_Disco(y_true, y_pred, Dphi, weight, lamb=0.1)
    return Disco

def Create_Model_with_Disco(input_shape):
    layer_opts = dict( activation = 'sigmoid', kernel_initializer = initializers.glorot_normal(seed=seed))
    Dphi_and_weight = Kl.Input(shape = int(2))
    input_layer = Kl.Input(shape = input_shape )
    x = Kl.Dense( 36, **layer_opts) (input_layer)
    x = Kl.Dense( 48, **layer_opts) (x)
    y_pred = Kl.Dense( 1., activation = 'sigmoid', name = "OutputLayer" )(x)
    model = Km.Model(inputs= [input_layer, Dphi_and_weight], outputs=y_pred )
    model_optimizer = Adam(lr=0.0001)
    model_disco = Loss_Disco(Dphi_and_weight, 0.1)
    model.compile(optimizer=tf.keras.optimizers.Adam(),loss=model_disco, metrics = ['accuracy'])
    model.summary()
    return model

def Train_NN(model, train, val, n_epochs = 400, batch_size = 2000,):
    fit_history_list = []
    fit_history_list.append(model.fit(train[0], train[1], epochs = n_epochs, shuffle = True, batch_size = batch_size, validation_data = (val[0],val[1]), sample_weight = train[2], callbacks=[tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20, verbose = True, min_delta = 0.001)])) #

    return fit_history_list


def Fold_Odd_Even(input, targets, weights, class_labels, Number):
    input_odd = input[Number % 2 == 1]
    targets_odd = targets[Number % 2 == 1]
    weights_odd = weights[Number % 2 == 1]
    class_labels_odd = class_labels[Number % 2 == 1]
    input_even = input[Number % 2 == 0]
    targets_even = targets[Number % 2 == 0]
    weights_even = weights[Number % 2 == 0]
    class_labels_even = class_labels[Number % 2 == 0]
    return (input_odd, targets_odd, weights_odd, class_labels_odd), (input_even, targets_even, weights_even, class_labels_even)


def Train_Odd_Even(odd, even, n_epochs = 400, batch_size = 2000):
    fit_history_list = []
    model_list = []
    input_shape = odd[0].shape[1]

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_step_decay)

    X_odd, X_val_odd, weights_odd, weights_val_odd, y_odd, y_val_odd, = train_test_split(odd[0], odd[2], odd[1], test_size=0.2)
    X_even, X_val_even, weights_even, weights_val_even, y_even, y_val_even, = train_test_split(even[0], even[2], even[1], test_size=0.2)

    model_odd = Create_Model_basic(input_shape)
    model_even = Create_Model_basic(input_shape)
    fit_history_odd = model_odd.fit(X_odd, y_odd, epochs = n_epochs, shuffle = True, batch_size = batch_size, validation_data=(X_val_odd, y_val_odd, weights_val_odd), sample_weight=weights_odd, verbose=0 ,callbacks=[tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 100, verbose = True, min_delta = 0.001),lr_schedule]) #
    fit_history_even = model_even.fit(X_even, y_even, epochs = n_epochs, shuffle = True, batch_size = batch_size, validation_data=(X_val_even, y_val_even, weights_val_even), sample_weight=weights_even, verbose=0 ,callbacks=[tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 100, verbose = True, min_delta = 0.001),lr_schedule]) #

    return fit_history_odd, model_odd, fit_history_even, model_even



def Train_NN_Kfold(train_data, val_data, n_epochs = 400, batch_size = 2000, num_folds = 2):
    fit_history_list = []
    model_list = []
    input_shape = train_data[0].shape[1]

    #Merge Data again
    input_scaled = np.concatenate((train_data[0], val_data[0]), axis=0)
    targets = np.concatenate((train_data[1], val_data[1]), axis=0)
    weights = np.concatenate((train_data[2], val_data[2]), axis=0)

    kfold = KFold(num_folds, True, 1)
    fold_no = 1

    for train, val in kfold.split(input_scaled, targets):
        train_data = input_scaled[train]
        val_data = input_scaled[val]
        train_label = targets[train]
        val_label = targets[val]
        train_weights = weights[train]
        val_weights = weights[val]

        #lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_step_decay)

        model = Create_Model_basic(input_shape)
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        fit_history = model.fit(train_data, train_label, epochs = n_epochs, shuffle = True, batch_size = batch_size ,validation_data = (val_data,val_label), sample_weight=train_weights, verbose=0 ,callbacks=[tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 50, verbose = True, min_delta = 0.001),lr_schedule]) #
        scores = model.evaluate(val_data, val_label, batch_size=batch_size, verbose=0)
        nn_scores = model.predict(val_data,verbose = True)

        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

        model_list.append(model)
        fit_history_list.append(fit_history)
        fold_no += 1

    return fit_history_list, model_list


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
