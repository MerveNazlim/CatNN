import os, sys, pickle, math, argparse
import h5py
import numpy as np

class Sample :

    """
    Sample

    This class will hold the feature data for a given sample.
    """

    def __init__(self, name = "", class_label = -1, input_data = None) :
        """
        Sample constructor

        Args :
            name : descriptive name of the sample (obtained from the input
                pre-processed file)
            input_data : numpy array of the data from the pre-processed file
                (expects an array of dtype = np.float64, not a structured array!)
            class_label : input class label as found in the input pre-processed
                file
        """

        if input_data.dtype != np.float64 :
            raise Exception("ERROR Sample input data must be type 'np.float64', input is '{}'".format(input_data.dtype))

        if class_label < 0 :
            raise ValueError("ERROR Sample (={})class label is not set (<0)".format(name, class_label))

        #print("Creating sample {} (label = {})".format(name, class_label))

        self._name = name
        self._class_label = class_label
        self._input_data = input_data
        self._regression_inputs = None

    def name(self) :
        return self._name
    def class_label(self) :
        return self._class_label
    def data(self) :
        return self._input_data
    @property
    def regression_inputs(self) :
        return self._regression_inputs
    @regression_inputs.setter
    def regression_inputs(self, data) :
        self._regression_inputs = data

def floatify(input_array, feature_list) :
    ftype = [(name, float) for name in feature_list]
    return input_array.astype(ftype).view(float).reshape(input_array.shape + (-1,))

def load_input_file(args, features_to_ignore) :

    """
    Check that the provided input HDF5 file is of the expected form
    as defined by the pre-processing. Exits if this is not the case.
    Returns a list of the sample names found in the file.

    Args :
        args : user input to the executable
    """

    # check that the file can be found
    if not os.path.isfile(args) :
        print("ERROR provided input file (={}) is not found or is not a regular file".format(args))
        sys.exit()

    samples_group_name = "samples"
    samples = []

    with h5py.File(args, 'r') as input_file :

        feature_list = list(input_file["samples"]["signal"]["features"].dtype.names)
        num_featurs = len(feature_list)




        for ignore_feature in features_to_ignore:
            print("Ignoring feature: ", ignore_feature)
            feature_list.remove(ignore_feature)

        print("Found {} features on the dataset {} features were loaded: \n".format(num_featurs-1, len(feature_list)-1))

        for feature in feature_list:
            if feature == "event_weight":
                print(" ")
                continue
            print(feature)

        # now build the samples
        class_names = []
        class_no = []
        if samples_group_name in input_file :
            sample_group = input_file[samples_group_name]
            for p in sample_group :
                class_names.append(p)
                process_group = sample_group[p]
                class_label = process_group.attrs['training_label']
                class_no.append(int(class_label))
                s = Sample(name = p, class_label = int(class_label), input_data = floatify( process_group['features'][tuple(feature_list)], feature_list ))
                samples.append(s)

        else :
            print("samples group (={}) not found in file".format(samples_group_name))
            sys.exit()

    samples = sorted(samples, key = lambda x: x.class_label())

    return samples, feature_list[:-1], class_names, class_no

def build_combined_input(training_samples) :

    targets = []
    # used extended slicing to partition arbitrary number of samples
    sample0, sample1, *other = training_samples

    targets.extend( np.ones( sample0.data().shape[0] ) * sample0.class_label() )
    targets.extend( np.ones( sample1.data().shape[0] ) * sample1.class_label() )

    inputs = np.concatenate( (sample0.data(), sample1.data()), axis = 0)
    for sample in other :
        inputs = np.concatenate( (inputs, sample.data()) , axis = 0 )
        targets.extend( np.ones( sample.data().shape[0] ) * sample.class_label() )

    targets = np.array(targets, dtype = int )
    # Extract weights and remove out of inputs
    weights = inputs[:,-1]
    inputs = inputs[:,0:-1]

    targets = targets[np.where((weights<1) &(weights>0))]
    inputs  = inputs[np.where((weights<1) &(weights>0))]
    weights = weights[np.where((weights<1) &(weights>0))]
    if targets[targets>1].size > 0:
        print("Dataset contains extra labels for different backgrounds!")
        for i in np.arange(targets.max()+1):
            print("Class", i," : ", targets[targets==i].size)
        class_labels = np.array(targets)
        targets[class_labels==0] = 1
        targets[class_labels>0] = 0
    else:
        class_labels = np.NaN

    print("Dataset contains {} Signal events and {} Background events.".format(targets[targets==1].size, targets[targets==0].size))

    return inputs, targets, class_labels, weights
