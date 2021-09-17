import matplotlib

matplotlib.use('TkAgg')
from fastai.vision.all import *  # important
import random
from datetime import datetime
from shutil import copyfile
import seaborn as sns
from . PlottingMetrics  import *
import argparse
import os
from . Train import *
import json

# https://forums.fast.ai/t/plotting-metrics-after-learning/69937/
# https://github.com/timeseriesAI/tsai

# Provide K-fold data keeping data in groups
class KFoldWithGroups:

    # Initlization of object
    # dataframe: input data
    # group_field: field name that groups the data
    # shuffle: True to shuffle the data, else otherwise
    # validate_nun_multiple: True to validate data non-multiple of k (last elements, after shuffle)
    # balance_label_field: Set field label, if the k-fold should balanced by the data label (k divisions for each label of data)
    #                      This option ensures that each training set will have at least 1 data of each label for highly unbalanced datasets
    def __init__(self, dataframe, k=2, group_field='name', shuffle=True, validate_non_multiple=True, balance_label_field=None):

        if shuffle:
            self.df = dataframe.sample(frac=1).reset_index(drop=True) # reset index?
        else:
            self.df = dataframe.copy()


        # TODO: Check that each group item belongs to the same label

        if balance_label_field is None:
            # Get data groups
            self.groups = self.df.groupby(group_field).head(1)[group_field].to_list()
            # Compute num groups per fold
            num_groups = len(self.groups)
            self.fold_num_groups = int(num_groups / k)
            # Check if k is too big
            if self.fold_num_groups == 0:
                raise RuntimeError("K is to big")
            self.balanced_by_label = False
        else:
            # Get data groups for each label
            self.groups = {key: v[group_field].to_list() for key, v in
                           self.df.groupby(group_field).head(1).groupby(balance_label_field)}
            # Compute num groups per fold for each label
            self.fold_num_groups = {}
            for key, items in self.groups.items():
                num_groups = len(self.groups[key])
                self.fold_num_groups[key] = int(max(num_groups / k, 1)) #TODO: should throw error?
                print(self.fold_num_groups[key])
            self.balanced_by_label = True

        self.k_idx = 0
        self.k = k
        self.group_field = group_field
        self.vnm = validate_non_multiple

    def GetKIteration(self, i=None):
        if i is None: #If k index is not specified, use the current one
            i = self.k_idx

        if i >= self.k or i < 0: #Check that index is inside the range
            #return None # Return none or throw error?
            raise RuntimeError("K index is out of range!")

        self.k_idx = i + 1 #Jump to next index, for next GetK run

        # Get groups for this fold
        if not self.balanced_by_label:  # If it is not balanced by error

            if self.vnm and i == self.k - 1:
                # if the index is the last, include non-multiple remaining cases at validation (ex: k2 for 5 cases: 0:[VVTTT] 1:[TTVVV])
                validate_groups = self.groups[i * self.fold_num_groups:]
                train_groups = self.groups[:i * self.fold_num_groups]
            else:
                validate_groups = self.groups[i * self.fold_num_groups:(i + 1) * self.fold_num_groups]
                train_groups = self.groups[:i * self.fold_num_groups] + self.groups[(i + 1) * self.fold_num_groups:]

        else:  #If groups should be balanced by label
            validate_groups = []
            train_groups = []
            for key, items in self.groups.items():
                if self.vnm and i == self.k - 1:
                    # if the index is the last, include non-multiple remining cases at validation (ex k2 for 5 cases: 0:[VVTTT] 1:[TTVVV])
                    validate_groups += self.groups[key][i * self.fold_num_groups[key]:]
                    train_groups += self.groups[key][:i * self.fold_num_groups[key]]
                else:
                    validate_groups += self.groups[key][i * self.fold_num_groups[key]:(i + 1) * self.fold_num_groups[key]]
                    train_groups += self.groups[key][:i * self.fold_num_groups[key]] + self.groups[key][(i + 1) * self.fold_num_groups[key]:]
                print(validate_groups)

        print("validate:", validate_groups)
        print("train:", train_groups)

        # Once the groups has been divided as valid or not,
        # know each image of the group is level accordignly

        # Create new dataframe and set valid all images of validate groups
        df_new = self.df.copy()
        # For each image set valid if its group field is inside validate_groups, otherwise is training
        df_new['is_valid'] = [item_name in validate_groups for item_name in self.df[self.group_field].to_list()]

        print("Training:", len(train_groups), " Total slides:", sum(df_new['is_valid'] == False))
        print("Validation:", len(validate_groups), " Total slides:", sum(df_new['is_valid'] == True))

        return df_new

    def Restart(self):
        self.k_idx = 0

    def GetDataFrame(self):
        return self.df.copy()


def TrainKFold(dataframe, k_folds, n_epochs, batch_size, lr, balance_training=True, output_folder = None,
               df_label='labels', df_filename='fname', df_group_field='name', files_path=""):

    output = {}

    # Create figure for input stats
    output["input_figure"], ax_input = plt.subplots(1, 3, figsize=(12,4))

    ax_input[0].set_title("Groups per label")
    sns.countplot(y=df_label, data=dataframe.groupby(df_group_field).head(1), ax=ax_input[0])
    ax_input[1].set_title("Images per label")
    sns.countplot(y=df_label, data=dataframe, ax=ax_input[1])
    ax_input[2].set_title("Images per group")
    sns.countplot(y=df_label, data=dataframe, hue=df_group_field, ax=ax_input[2])
    ax_input[2].get_legend().remove()


    k_output_folder = None
    if output_folder:
        if os.path.exists(output_folder):
            print("Error: Output folder already exists!")
        else:
            os.mkdir(output_folder)
            output["input_figure"].savefig(os.path.join(output_folder, "input.png"))


    # Create a K-fold generator for this dataframe
    df_k_generator = KFoldWithGroups(dataframe, k=k_folds, group_field=df_group_field,
                                     shuffle=False, balance_label_field=df_label)

    eval_dataset = []
    targs = []
    preds = []

    # For each fold
    for k_id in range(k_folds):

        print("{}-fold iteration {}".format(k_folds, k_id))

        # Get this fold dataframe
        df_k = df_k_generator.GetKIteration(k_id)

        if output_folder:
            k_output_folder = os.path.join( output_folder, str(k_id) )

        output[k_id] = Train(df_k, n_epochs, batch_size, lr, balance_training=balance_training, evaluate=True,
                             output_folder = k_output_folder, df_label=df_label, df_filename=df_filename,
                             df_group_field=df_group_field, files_path=files_path)

        eval_dataset .append(output[k_id]["evaluate"]["dataset"])
        preds.append(output[k_id]["evaluate"]["preds"])
        targs.append(output[k_id]["evaluate"]["targs"])

    output["evaluate"] = {}
    output["evaluate"]["dataset"] = df_k_generator.GetDataFrame()
    output["evaluate"]["preds"] = np.concatenate(preds)
    output["evaluate"]["targs"] = np.concatenate(targs)
    output["evaluate"]["roc"], output["evaluate"]["roc_auc"] = PlotRoc(output["evaluate"]["preds"], output["evaluate"]["targs"])


    if output_folder:
            output["evaluate"]["roc"].savefig(os.path.join(output_folder, "roc.png"))


    return output