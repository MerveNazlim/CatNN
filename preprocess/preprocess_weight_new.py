#!/usr/bin/env python
import argparse
import sys
import os
import glob
import json
import hashlib
from time import time

# h5py
import h5py
# numpy
import numpy as np
from numpy.lib.recfunctions import append_fields

class FileInput :
    """
    FileInput structure

    This structure is an interface to the input samples and should
    encode things like the associated training label for this sample,
    the input filepath, and the descriptive name of the sample.

    Note:
        Currently this structure can only be built from the text file format
        used in initial builds.

    Args:
        descriptor : one of the lines in an input text filelist, expected to be of
            CSV with 3 columns: 1) training label (integer), 2) descriptive name (one-word),
            3) full path to this sample's file
    Attributes:
        label : training label for this sample
        name : descriptive name for this sample (one word)
        filepath : full path to the original input HDF5 file
        descriptor : input text file descriptor

    """
    def __init__(self, descriptor = "") :
        self._label = -1
        self._name = ""
        self._filepath = ""
        self._raw_descriptor = ""

        self.descriptor = descriptor

    @property
    def descriptor(self) :
        return self._raw_descriptor
    @descriptor.setter
    def descriptor(self, raw_descriptor = "") :
        self._raw_descriptor = raw_descriptor
        d = raw_descriptor.strip()
        d = d.split()
        if len(d) != 3 :
            raise Exception("input descriptor not of expected form (descriptor is : {})".format(raw_descriptor))
        self.label = int(d[0])
        self.name = str(d[1])
        if not (str(d[2]).endswith(".h5") or str(d[2]).endswith(".hdf5")) :
            raise Exception("an input file does not appear to be an HDF5 file (file : {})".format(str(d[2])))
        self.filepath = str(d[2])

    @property
    def label(self) :
        return self._label
    @label.setter
    def label(self, val) :
        self._label = val

    @property
    def name(self) :
        return self._name
    @name.setter
    def name(self, val) :
        self._name = val

    @property
    def filepath(self) :
        return self._filepath
    @filepath.setter
    def filepath(self, val) :
        self._filepath = val

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

def floatify(input_array, feature_list) :
    ftype = [(name, float) for name in feature_list]
    return input_array.astype(ftype).view(float).reshape(input_array.shape + (-1,))

def unique_filename(file_name) :

    if os.path.isfile(file_name):
        expand = 1
        while True:
            expand += 1
            new_file_name = file_name.split(".h5")[0] + "_" + str(expand) + ".h5"
            if os.path.isfile(new_file_name):
                continue
            else:
                file_name = new_file_name
                break
    return file_name

def inputs_from_text_file(text_file = "") :

    """
    Return a list of FileInput objects for each of the lines in
    the input text file. Each line of the input file is assumed to
    be 3 columns:
        col 0 : training label (integer)
        col 1 : name of sample (string)
        col 2 : full path to input HDF5 file

    Arguments:
        text_file : input *.txt file
    """

    if text_file == "" :
        raise Exception("input text file is an empty string")

    lines = [l.strip() for l in open(text_file).readlines()]
    out = []
    for l in lines :
        if not l : continue
        if l.startswith("#") : continue
        fi = FileInput(l)
        out.append(fi)
    return out

def inputs_from_dir(filedir) :

    """
    Return a list of *.hdf5 or *.h5 files contained in
    the provided directory

    Arguments:
        filedir : directory with *.hdf5 or *.hf files inside
    """

    dir_files = glob.glob("{}/*.hdf5".format(filedir))
    dir_files += glob.glob("{}/*.h5".format(filedir))
    return dir_files

def fields_represented(required_fields = [], sample_fields = []) :

    """
    Determine if all entries in an input list of required fields
    are in the list of fields built from an input file

    Arguments:
        required_fields : the list of fields (variables) that we want
        sample_fields : the list of fields (variables) that are currently in the file
    """

    return set(required_fields).issubset(sample_fields)

def datasets_with_name(input_files = [], dataset_name = "", req_fields = None) :

    """
    From an input list of HDF5 files, return a list of such files
    that contain a given top-level dataset node name and
    a set of fields

    Arguments:
        input_files : input list of HDF5 files
        dataset_name : name of dataset that is required to be in the files
        req_fields : HDF5 fields that must be in the dataset (if passing all, then
                        this string is expected to be a string "ALL_FIELDS"
    """

    out = []
    for ifile in input_files :
        with h5py.File(ifile.filepath, 'r') as sample_file :
          #  import pdb; pdb.set_trace()
            if dataset_name in sample_file :
                dtype = sample_file[dataset_name].dtype.names
                sample_fields = list(sample_file[dataset_name].dtype.names)
                if req_fields != "ALL_FIELDS" and len(req_fields) > 0 :
                    if not fields_represented(req_fields, sample_fields) :
                        print("WARNING dataset (={}) does not have all of the required fields, missing {}".format(ifile.filepath, set(sample_fields) - set(req_fields)))
                        continue
                out.append(ifile)
            else :
                print("WARNING dataset (={}) not found in input file {}".format( dataset_name, ifile.filepath ))
    return out

def get_inputs(args) :

    """
    Return a list of *.hdf5 or *.h5 files that are located
    in the user-provided input. Only builds a list of those
    files that contain the user-provided top-level group.
    """

    user_input = args.input
    dsname = args.dataset_name
    requested_fields = get_features(args)
    if requested_fields == "all" :
        requested_fields = "ALL_FIELDS"

    # text file with multiple files
    if os.path.isfile(user_input) and user_input.endswith(".txt") :
        return datasets_with_name(input_files = inputs_from_text_file(user_input), dataset_name = dsname, req_fields = requested_fields)

def features_from_file(input_file) :

    """
    From an input text file that has a single feature (variable)
    on each line, return a list of those features

    Args:
        input_file : input text file that contains a single feature (variable)
            on each line
    """

    out = []
    lines = [l.strip() for l in open(input_file)]
    for l in lines :
        if not l : continue
        if l.startswith("#") : continue
        out.append(l)
    return out

def get_features(args) :

    """
    Return a list of the features (variables) to slim on.
    """

    if args.feature_list == "all" :
        return []

    if os.path.isfile(args.feature_list) :
        return features_from_file(args.feature_list)
    else :
        return args.feature_list.split(",")

def preprocess_file(input_file, input_group, args) :

    feature_list = get_features(args)
    if feature_list == "all" :
        feature_list = []

    with h5py.File(input_file.filepath, 'r') as infile :
        ds = infile[args.dataset_name]

        #Clean up remove events with NANs, negativ weights, or infinit weights
        indices = ~np.isnan(ds['totalEventsWeighted'])
        ds = ds[indices]
        indices = (ds['totalEventsWeighted'] > 0)
        ds = ds[indices]
        indices = np.isfinite(ds['totalEventsWeighted'])
        ds = ds[indices]


        asd= (36207.66*(ds['RunYear'][:]==2015)+36207.66*(ds['RunYear'][:]==2016)+44307.4*(ds['RunYear'][:]==2017)+58450.1*(ds['RunYear'][:]==2018))*ds['weight_pileup'][:]*ds['jvtSF_customOR'][:]*ds['bTagSF_weight_DL1r_77'][:]*ds['weight_mc'][:]*ds['xs'][:]/ds['totalEventsWeighted'][:]
        ds = append_fields(ds, 'event_weight', asd,  '<f4')

        #Clean up remove events with NANs, negativ weights, or infinit weights
        feature_list.append('event_weight')
        indices = ~np.isnan(ds['event_weight'])
        ds = ds[indices]
        indices = (ds['event_weight'] > 0)
        ds = ds[indices]
        indices = np.isfinite(ds['event_weight'])
        ds = ds[indices]

        if len(feature_list) > 0 :
            ds = ds[feature_list]

        np.random.shuffle(ds)

        out_ds_train = input_group.create_dataset("features", shape = ds.shape, dtype = ds.dtype, data = ds, maxshape = (None,))

def preprocess(inputs, args) :

    output_filename = ""
    if args.outdir != "" :
        mkdir_p(args.outdir)
    output_filename += args.outdir
    output_filename += "/{}".format(args.output)
    output_filename = unique_filename(output_filename)

    with h5py.File(output_filename, "w") as outfile :

        sample_group = outfile.create_group("samples")

        for i, input_file in enumerate(inputs) :

            input_group = sample_group.create_group(input_file.name)
            input_group.attrs['training_label'] = input_file.label
            input_group.attrs['filepath'] = input_file.filepath
            print(input_file.label)
            print(input_file.name)
            preprocess_file(input_file, input_group, args)

    print("storing preprocessed file at : {}".format(os.path.abspath(output_filename)))

def main() :

    parser = argparse.ArgumentParser(description = "Pre-process your inputs")
    parser.add_argument("-i", "--input", help = "Provide input HDF5 files [HDF5 file, text filelist, or directory of files]", required = True)
    parser.add_argument("-o", "--output", help = "Provide output filename", default = "2hdm_signals.h5")
    parser.add_argument("--outdir", help = "Provide an output directory for file dumps [default: ./]", default = "./")
    parser.add_argument("-f", "--feature-list", help = "Provide list of features to slim on [comma-separated list, text file]", default = "all")
    parser.add_argument("-d", "--dataset-name", help = "Common dataset name in files", default = "nominal")
    parser.add_argument("-s", "--selection-file", help = "Provide a selection file (JSON)", default = "")
    parser.add_argument("-v", "--verbose", help = "Be loud about it", default = False, action = 'store_true')
    args = parser.parse_args()

    if args.selection_file != "" :
        # right now just hard code the selection
        print("Selection file handling not yet implemented, exiting")
        sys.exit()

    inputs = get_inputs(args)
    print("Found {} input files".format(len(inputs)))

    preprocess(inputs, args)

#_________________________________
if __name__ == "__main__" :
    main()
