import math
import json
import os.path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ICEBERG3DLoader import Loaders, Subtype
from . SliceTumor import *
from . config import *
import cv2

# Status function
def Status(v, total, text="", bars=20, single_line=False):
    bar = ("{:<" + str(bars) + "}").format("#" * int(bars * v / total))
    status_text = "Status: [{}] {:>3}%  ({:>" + str(len(str(total))) + "}/{})  {}" #"Status: [{}] {:>3}%  ({:>6}/{:<6})   {}" fixed

# Clock class
from time import time
class Clock():
    def __init__(self):
        self.start = time()
    def GetSeconds(self):
        return time() - self.start
    def Took(self):
        seconds_elapsed = self.GetSeconds()
        print("It took {:.0f}h {:.0f}m {:.0f}s".format(seconds_elapsed/3600, (seconds_elapsed/60) % 60, seconds_elapsed % 60))


# Load dataset and subtype
def LoadData( options_dict ):

    options = default_options.copy()
    options.update(options_dict)

    if options['dataset'] not in datasets:
        raise RuntimeError("Dataset {} not found!".format(options['dataset']))

    # Folders
    options['dataset_folder'] = datasets[options['dataset']]['dataset_folder']
    options['data_folder'] = datasets[options['dataset']]['data_folder']
    options['data_gt'] = datasets[options['dataset']]['data_gt']
    options['subtype_csv'] = datasets[options['dataset']]['subtype_csv']

    if len(options['class_labels']) != 1:
        raise RuntimeError("Unsupported function: currently only one subtype can be classified")

    # Load Dataset
    print("Loading {} dataset...".format(options['dataset'].upper()))
    data = Loaders.LoadDataset(options, None, load_only=True)  # True
    print("Tumor cases: {}".format(len(data[0])))

    print("Loading subtype..")
    df_subtype = Subtype.SubtypeData(options['dataset'], options['subtype_csv'])

    print("Getting <{}>..".format(options['class_labels']))
    df = df_subtype[options['class_labels']]

    print("Filtering unclear data...")
    df = Subtype.FilterUnclearLabel(df, options['class_labels'])
    print(("Unclear cases: {}".format(len(df_subtype) - len(df))))
    print("Subtype entries: {}".format(len(df)))

    # print("Sample:")
    # print(df.head())

    output = options.copy()
    # Show subtype stadistics
    #output["stats_fig"], ax = plt.subplots(1, 1, figsize=(10, 10))
    # sns.countplot(y='Pam50.Call',data=df, hue="Pam50.Call",ax=ax[0])
    #sns.countplot(y=options['class_labels'][0], data=df, hue=options['class_labels'][0], ax=ax)

    output["stats"] = {class_label: df[class_label].value_counts().to_dict() for class_label in
                        options['class_labels']}

    return data, df, output



# Generate all slices for a dataset
def SliceGeneration(options_dict):

    options = default_options.copy()
    options.update(options_dict)

    # Load data
    (pres, posts, mask_labels), df, output = LoadData(options)

    # Slice folder name
    if options['slice_folder'] is None:
        options['slice_folder'] = "./{}_{}_{:03.0f}".format(options['dataset'], options['class_labels'][0],
                                                            options['percentage_slices'] * 100)

    # Create output slice folder
    print("Creating output folder <{}>...".format(options['slice_folder']))
    if os.path.exists(options['slice_folder']):
        raise RuntimeError("Output folder already exists!")
    os.mkdir(options['slice_folder'])

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    dataframe = {"fname": [], "labels": [], "name": [], "slice": []}  # valid

    # Set slice parameters
    slice_axis = options['slice_axis']
    percentage_slices = options['percentage_slices']

    clock = Clock()


    fnames = []
    labels = []
    names = []
    slices = []

    print("Processing tumors...")

    # For each tumor of the dataset
    for index in range(len(pres)):

        pre, post, mask = pres[index], posts[index], mask_labels[index]

        # Get name
        name = os.path.basename(os.path.dirname(pre))

        print(" {:>4}/{:<4}: {}...".format(index, len(pres), name))

        # Check if the subtype info exists for the specified case
        if name not in df.index.values:
            print("\t  {:>6}: {}".format(name, "Error: ubtype not found for <{}>".format(name)))
            continue

        # Get subtype info
        class_type = str(df.at[name, options['class_labels'][0]])

        # generate tumor slice and save image
        tumor_result = SliceTumor(  name, pre, post, mask, options['input_format'], class_type,
                                    options['slice_folder'], percentage_slices=percentage_slices,
                                    slice_axis=slice_axis)

        # Save tumor slice information
        fnames += tumor_result[0]
        labels += tumor_result[1]
        names += tumor_result[2]
        slices += tumor_result[3]

        # Show info/errors
        for index, msg in tumor_result[4]:
            print("\t  {:>6}: {}".format(index, msg))


    # generate label table
    print("")
    print("Generating list of files...")
    import itertools
    dataframe["fname"] = list(itertools.chain(*fnames))
    dataframe["labels"] = list(itertools.chain(*labels))
    dataframe["name"] = list(itertools.chain(*names))
    dataframe["slice"] = list(itertools.chain(*slices))

    #Save table in a csv file
    print("Saving CSV data..")
    df = pd.DataFrame(data=dataframe)
    df.to_csv(os.path.join(options['slice_folder'], "train.csv"), index=False)
    # print(df)

    # Save output options into a json
    print("Saving info...")
    with open(os.path.join(options['slice_folder'], "info.json"), 'w') as fp:
        json.dump(output, fp)

    clock.Took()
    #  cv2.destroyAllWindows()






# Same than SliceGeneration but using multithrads
def SliceGenerationMultiThread(options_dict, num_threads = 10):

    options = default_options.copy()
    options.update(options_dict)

    # Load data
    (pres, posts, mask_labels), df, output = LoadData(options)

    # Slice folder name
    if options['slice_folder'] is None:
        options['slice_folder'] = "./{}_{}_{:03.0f}".format(options['dataset'], options['class_labels'][0], options['percentage_slices'] * 100)

    # Create output slice folder
    print("Creating output folder <{}>...".format(options['slice_folder']))
    if os.path.exists(options['slice_folder']):
        raise RuntimeError("Output folder already exists!")
    os.mkdir(options['slice_folder'])

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    dataframe = {"fname": [], "labels": [], "name": [], "slice": []}  # valid

    # Set slice parameters
    slice_axis = options['slice_axis']
    percentage_slices = options['percentage_slices']

    # Define a multi-thread function for a single tumor (Multi-thread function)
    global ProcessTumorMT
    def ProcessTumorMT(index):

        pre, post, mask = pres[index], posts[index], mask_labels[index]

        # Get name
        name = os.path.basename(os.path.dirname(pre))

        print("#", end="", flush=True)

        # Check if the subtype info exists for the specified case
        if name not in df.index.values:
            info = (name, "Error: ubtype not found for <{}>".format(name))
            return [], [], [], [], info

        # Get subtype info
        class_type = str(df.at[name, options['class_labels'][0]])

        return SliceTumor(  name, pre, post, mask, options['input_format'], class_type, options['slice_folder'],
                            percentage_slices=percentage_slices, slice_axis=slice_axis)
    #########

    clock = Clock()

    print("Processing tumors... ({} threads)".format(num_threads))
    import multiprocessing
    pool = multiprocessing.Pool(num_threads)
    fnames, labels, names, slices, info = zip(*pool.map(ProcessTumorMT, range(len(pres))))

    print("")
    print("Generating list of files...")
    import itertools
    dataframe["fname"] = list(itertools.chain(*fnames))
    dataframe["labels"] = list(itertools.chain(*labels))
    dataframe["name"] = list(itertools.chain(*names))
    dataframe["slice"] = list(itertools.chain(*slices))
    info = list(itertools.chain(*info))

    if len(info) > 0:
        print("Problems:")
        for index, msg in info:
            print("  {:>6}: {} ".format(index, msg))
    else:
        print("No problems found.")


    print("Saving CSV data..")
    df = pd.DataFrame(data=dataframe)
    df.to_csv(os.path.join(options['slice_folder'], "train.csv"), index=False)
    #print(df)

    print("Saving info...")
    with open(os.path.join(options['slice_folder'], "info.json"), 'w') as fp:
        json.dump(output, fp)

    clock.Took()
    #  cv2.destroyAllWindows()



