# Dataset Tools for TCGA-BRCA
# Author: Joel Vidal <jolvid@gmail.com>
# Date: 20/04/20
# Version: 0.4

import numpy as np
from . LoadDicomSeries import *
from . MRIOrientation import TransformMRIOrientation, GetAnatomicalOrientationFromOrientation, CheckValidOrientation
from . BreastCrop import GetBreastCropedRegion
import os
import itertools
import SimpleITK as sitk
import random
import struct
import json

#TODO: Add resampling function
#      Remove post index from name and check if they exist instead?
#      Add support for A0, AR
#      Add/Improve comments

# MAIN FUNCTIONS ---

"""
This function format the dataset and gt .les files, save it into NIFTI format and load the dataset
If a dataset name is not provided, a unique name is used for different configurations
NOTICE: AO and AR not fully implemented/tested yet

Parameters:
 - dataset_folder: Folder where original dataset is saved
 - les_folder: Folder where .les ground truth files are saved
 - output_folder: Folder where the output formated dataset will be saved
 - crop_breast: Crop the breast region of the dataset (Not tested)
 - orientation: Set the desired direction matrix or anatomical orientation of the patient
 - center: Recenter the MRI at 0,0,0
 - bias_fiel_n4: Apply the bias field n4 to all dataset images
 - save_metadata: Save metadata json files
 - serie_list: list of series to be loaded (if found in dataset)
 - post_subset: list of post image to add in the dataset
 - force_rewrtie: force the rewrite of all dataset files
 - dataset_name: Set a custom name for the dataset
"""
def AutoFormatDataset(dataset_folder, les_folder, output_folder, crop_breast=False, orientation=None, center=False,
                      bias_field_n4=False, save_metadata=True, serie_list=["AO", "AR", "BH", "E2"], post_subset=None,
                      force_rewrite=True, dataset_name=None):

    # Check input folder
    if not os.path.exists(dataset_folder):
        raise RuntimeError("Input dataset folder do not exists!")
    if not os.path.exists(les_folder):
        raise RuntimeError("Input les files folder do not exists!")

    # Create output folder
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    gt_folder = os.path.join(output_folder, "ground_truth")

    # Create gt
    CreateGTs(dataset_folder, les_folder, gt_folder, serie_list, force_rewrite)

    if type(dataset_name) is not str:
        dataset_name = GetDatasetName(crop_breast, orientation, center, bias_field_n4, post_subset)

    formated_folder = os.path.join(output_folder, dataset_name)

    # Create Formated dataset
    CreateFormatedDataset(dataset_folder, gt_folder, formated_folder, crop_breast, orientation, center, bias_field_n4,
                          save_metadata, serie_list, post_subset, force_rewrite)

    print("\n> Output dataset folder: {}".format(formated_folder))

    #Load dataset
    dataset = LoadFromattedDataset(formated_folder, serie_list)

    return dataset

    #return formated_folder


# Get the Pre, Post/s and Labels from dataset
def FormatFilesLists( dataset, post_idx = 0, shuffle = True, maximum_files=None, serie_list = None ):

    image_pre_list = []
    image_post_list = []
    label_list = []
    name_list = []
    for key, item in dataset.items():
        # print(item)
        # Filter by serie
        if type(serie_list) is list and item["serie"] not in serie_list:
            continue

        # Get label
        label_list.append(item["gt"])

        # Get pre
        image_pre_list.append(item["pre"][0])

        # Get post
        if not isinstance(post_idx, list) or not isinstance(post_idx, tuple):
            if post_idx <= -1: #all elements
                image_post_list.append(tuple(item["post"]))
            else:
                item_post = [pi for pi in item["post"] if pi.endswith("_POST_{}.nii.gz".format(post_idx+1))]
                if len(item_post) != 1:
                    raise RuntimeError("Post image not found. Post index parameter out of range!")
                image_post_list.append(item_post[0])

        else:
        #Get multiple posts
            item_post = []
            for p_idx in post_idx:
                item_post += [pi for pi in item["post"] if pi.endswith("_POST_{}.nii.gz".format(p_idx+1))]
            if len(item_post) != len(post_idx):
                raise RuntimeError("Post image not found. Post index parameter out of range!")
            image_post_list.append(tuple(item_post))


    if shuffle: #List is added for support with python3
        lists_ziped = list(zip(image_pre_list, image_post_list, label_list))  # zip
        random.shuffle(lists_ziped)  # shuffle
        image_pre_list, image_post_list, label_list = zip(*lists_ziped)  # unzip
        image_pre_list, image_post_list, label_list = list(image_pre_list), list(image_post_list), list(label_list)

    if type(maximum_files) is not None:
        image_pre_list, image_post_list, label_list = image_pre_list[:maximum_files], image_post_list[:maximum_files], label_list[:maximum_files]

    return image_pre_list, image_post_list, label_list



# SPLIT TOOLS -----

# NOTICE: For additional options check for scikit-learn already data split functions
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection

# TODO: Not tested
def Shuffle(arrays):
    if type(arrays) is not list and type(arrays) is not tuple:
        raise RuntimeError("Unexpected array format")

    if type(arrays[0]) is not list and type(arrays[0]) is not tuple:
        arrays = [arrays]

    lists_ziped = list(zip(*arrays))  # zip
    random.shuffle(lists_ziped)  # shuffle
    result = zip(*lists_ziped)  # unzip
    result = [list(res) for res in result]
    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)

# Split a set of arrays into fractions
# Ex:   [arr_part1, arr_part2] = Split(arr, 0.5)
# Ex:   [arr1_part1, arr1_part2], [arr2_part1, arr2_part2], etc = Split([arr1, arr2, etc], 0.5)
def Split(arrays, split_fraction = 0.5 ):
    if type(arrays) is not list and type(arrays) is not tuple :
        raise RuntimeError("Unexpected array format")

    if type(arrays[0]) is not list and type(arrays[0]) is not tuple:
        arrays = [arrays]

    result = []
    for arr in arrays:
        half = int(float(len(arr)) * split_fraction)
        result.append([arr[:half], arr[half:]])

    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)



# Split a set of arrays into K-Folds
# Check KFold or KFoldIterator for a simpler option
# Ex: [arr1_1,..., arr1_k], ... [arrn_1,  ..., arrn_k] = KFoldSplit([arr1,..,arrn], k)
#     For fold 2:    train = arr_1 + arr_3 + ... + arr_n    test = arr_2
def KFoldSplit(arrays, k ):
    if type(arrays) is not list and type(arrays) is not tuple :
        raise RuntimeError("Unexpected array format")

    if type(arrays[0]) is not list and type(arrays[0]) is not tuple:
        arrays = [arrays]

    if k < 2:
        raise RuntimeError("k out of range [2,n]")

    if any(len(arr) < k for arr in arrays):
        raise RuntimeError("An array is too short for a {}-fold".format(k))

    result = []
    for arr in arrays:
        size_fold = len(arr) // k
        folded = [arr[size_fold*i:size_fold*(i+1)] if (i+1) < k else arr[size_fold*i:] for i in range(k)]
        result.append(folded)

    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)



# Split and interate a set of arrays into consecutive k-folds
# Check KFold or KFoldIterator for a simpler option
#
# Ex: for fold in KFoldIterator(arr, k)
#          train_arr1, test_arr1 = fold
#
# Ex: for fold in KFoldIterator([arr1,..,arrn], k)
#          train_arr1, test_arr1 = fold[arr1]
#          ...
#          train_arrn, test_arrn = fold[arrn]
class KFoldIterator:

    def __init__(self, arrays, k ):

        if type(arrays) is not list and type(arrays) is not tuple:
            raise RuntimeError("Unexpected array format")

        if type(arrays[0]) is not list and type(arrays[0]) is not tuple:
            arrays = [arrays]

        if k < 2:
            raise RuntimeError("k out of range [2,n]")

        if any(len(arr) < k for arr in arrays):
            raise RuntimeError("An array is too short for a {}-fold".format(k))

        self.k = k
        self.list_kfold_arrays = KFoldSplit(arrays, k)

    def __getitem__(self, index):
        data = [ (list(itertools.chain.from_iterable([arr[i] for i, fold in enumerate(arr) if i != index])), arr[index] ) for arr in self.list_kfold_arrays ]
        if len(data) == 1:
            return data[0]
        else:
            return tuple(data)

    def __len__(self):
        return self.k


# Split a set of arrays into an specific iteration of a K-fold
# EX:  for it in range(k):
#         train_arr, test_arr = KFold(arr, k ,it)
#
# EX:  for it in range(k):
#         [train_arr1, test_arr1], ..., [train_arrn, test_arrn] = KFold([arr1, ... ,arrn], k ,it)
def KFold(arrays, k, it ):
    if k < 2:
        raise RuntimeError("k out of range [2,n]")

    if it >= k or it < 0:
        raise RuntimeError("Iteration out of range [0,k)")

    if type(arrays) is not list and type(arrays) is not tuple :
        raise RuntimeError("Unexpected array format")

    if type(arrays[0]) is not list and type(arrays[0]) is not tuple:
        arrays = [arrays]

    if any(len(arr) < k for arr in arrays):
        raise RuntimeError("An array is too short for a {}-fold".format(k))

    result = []
    for arr in arrays:
        size_fold = len(arr) // k
        #split = [arr[size_fold*i:size_fold*(i+1)] if (i+1) < k else arr[size_fold*i:] for i in range(k)]
        test = arr[size_fold*it:size_fold*(it+1)] if (it+1) < k else arr[size_fold*it:]
        #https://stackoverflow.com/questions/1720421/how-do-i-concatenate-two-lists-in-python
        train = list(itertools.chain.from_iterable([arr[size_fold*i:size_fold*(i+1)] if (i+1) < k else arr[size_fold*i:] for i in range(k) if i != it]))
        result.append((train, test))

    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)


# Create a dictionary of a list of arrays (by number or names list)
def DictList(arrays, names=None):
    if type(arrays) is not list and type(arrays) is not tuple :
        raise RuntimeError("Unexpected array format")

    if type(arrays[0]) is not list and type(arrays[0]) is not tuple:
        arrays = [arrays]

    if type(names) is not list:
        return {idx: list(items) for idx, items in enumerate(zip(*arrays))}
    elif len(names) == len(arrays[0]) and all(type(n) is str for n in names) and all(names[0] != names[i] for i in range(1, len(names))):
        return {names[idx]: list(items) for idx, items in enumerate(zip(*arrays))}
    else:
        raise RuntimeError("Unexptected names format!")




# LOAD AND FORMAT DATASET -----

# Load all filenames of a formatted dataset in a dictonary
# - formated_dataset_folder: folder where the formatted dataset is saved
# - serie_list: list of series to be loaded (if found in dataset)
def LoadFromattedDataset(formatted_dataset_folder, serie_list=["AO", "AR", "BH", "E2"]):
    result = {}

    for ref in os.listdir(formatted_dataset_folder):

        folder = os.path.join(formatted_dataset_folder, ref)

        if len(ref.split("-")) < 3:
            print("Folder {} unkown".format(ref))
            continue

        dataset = ref.split("-")[0]
        serie = ref.split("-")[1]
        patient = ref.split("-")[2]

        if type(serie_list) is list and serie not in serie_list:
            continue

        list_nifti = [i for i in os.listdir(folder) if i.endswith(".nii.gz")]

        gt_filename = os.path.join(folder, "gt.nii.gz")
        if not os.path.exists(gt_filename):
            raise RuntimeError("GT missing! {}".format(gt_filename))

        pre_filenames = [os.path.join(folder, file) for file in list_nifti if file.endswith("PRE_1.nii.gz")]
        if len(pre_filenames) != 1:
            raise RuntimeError("Pre file missing! {}".format(folder))

        post_filenames = [os.path.join(folder, file)  for file in list_nifti if "_POST_" in file]
        if len(post_filenames) < 3:
            raise RuntimeError("POST file missing! {}".format(folder))

        # CHECK and SORT POST
        post_filenames_sorted = []
        for i in range(1, len(post_filenames) + 1):
            candidates = [p for p in post_filenames if "_POST_{}.nii.gz".format(i) in p]
            if len(candidates) != 1:
                raise RuntimeError("Unexpected number of candidates for post case")
            post_filenames_sorted.append(candidates[0])

        result[ref] = {
            "gt": gt_filename,
            "pre":  pre_filenames,
            "post": post_filenames_sorted,
            "serie": serie,
            "patient": patient
        }

    return result



#Auto generate a dataset name based on the options
#NOTICE: post_subset generates a different folder due to the fact that with the current code
#        it will require force_rewrite to add the new post in the same folder
#        todo: Change this
def GetDatasetName(crop_breast, orientation, center, bias_field_n4, post_subset):

    if type(post_subset) is list:
        if max(post_subset) < 16:
            post_subset_code = sum([pow(2, i - 1) for i in post_subset])
        else:
            print("Warning: Generic name used...")
            post_subset_code = "#"
    else:
        post_subset_code = 0

    dataset_name = "dataset_{}{}{}{}".format(1 if crop_breast else 0, 1 if center else 0,
                                                1 if bias_field_n4 else 0, post_subset_code)

    if orientation is not None:
        orientation = GetAnatomicalOrientationFromOrientation(orientation)
        dataset_name += "_" + orientation

    return dataset_name


# Get size of a file given number of components
# NOTE: Number of components is requried due to the fact that some DICOM files do not include the slide location tag
# The new LoadDicom v0.3 include a possible fix but it has not been fully tested yet
def GetSize(file, num_components=0):
    if not os.path.exists(file):
        print("ERROR: %s do not exists!" % (file))
        exit(-1)

    if os.path.isdir(file):
        org_image, num_components = LoadDicomSeries(file, None, True, num_components, verbose=2)  # num_component
        # org_image, num_comp = LoadDicomSeries(file, None, True, 0, -1, 2) #num_components
        # if num_components != num_comp:
        #    print("WRONG {}  ({}-{})".format(file, num_comp, num_components))

    else:
        org_image = sitk.ReadImage(file)
        num_components = org_image.GetNumberOfComponentsPerPixel()

    # print(file, org_image.GetDirection())
    org_image_size = list(org_image.GetSize())
    if org_image.GetNumberOfComponentsPerPixel() != num_components:
        print("[WARNING] Missmatching number of components ({}) set to {} File:{}".format(
            org_image.GetNumberOfComponentsPerPixel(), num_components, file))
        org_image_size[2] = (org_image_size[2] * org_image.GetNumberOfComponentsPerPixel()) // num_components

    return tuple(org_image_size)




# Basic processing of sitk images
# NOTE: Some options are experimental...
def ProcessImage(image, orientation=None, bias_field_n4=False,
                 force_center=False, crop_region=None):
    # TODO: Manage the metadata json here

    if crop_region is not None:
        if not isinstance(crop_region, (list, tuple)) or len(crop_region) != 6:
            raise RuntimeError("Unexpected crop region parameter")
        image = image[crop_region[0]:crop_region[1], crop_region[2]:crop_region[3],crop_region[4]:crop_region[5]]

    if orientation is not None:
        image = TransformMRIOrientation(image, orientation, verbose=False)

    if force_center:
        image.SetOrigin([0, 0, 0])

    if bias_field_n4:
        #https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1N4BiasFieldCorrectionImageFilter.html
        # maskImage = sitk.ReadImage(sys.argv[4])
        maskImage = sitk.OtsuThreshold(image, 0, 1, 200)
        image = sitk.Cast(image, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        numberFittingLevels = 4
        maxIterationsPerLevel = 50  # Default 50
        corrector.SetMaximumNumberOfIterations([int(maxIterationsPerLevel)] * numberFittingLevels)
        image = corrector.Execute(image, maskImage)

    return image




# Create a formated dataset with preprocessed data and GT files in NIFTI format (NIFTI GTs are required)
def CreateFormatedDataset(dataset_folder, gt_folder, output_folder, crop_breast=False, orientation=None, center=False,
                          bias_field_n4=False, save_metadata=True, serie_list=["AO", "AR", "BH", "E2"], post_subset=None,
                          force_rewrite=False):
    # TODO: ADD verbose
    # TODO: Allow to save all components in a single file
    # TODO: Save parameters in a dicctionari and save the file in json to identify the dataset( Load the dataset could be as easy as load the json )



    # Special cases dictionary
    discard_unknown_exceptions = True
    exceptions_list = {
        # "Patient":(name pre, name post)
        # "A1LG": difficult case, need cheking which two to pick
    }

    #Additional options
    save_error_gt_cases = False #if get is los save folder with name "error_"

    # Check if orientation set is valid:
    if orientation is not None:
        try: #This code also checks the orientation
            orientation_str = GetAnatomicalOrientationFromOrientation(orientation)
        except  Exception as e:
            print("Orientation problem. " + str(e))

    # Create output folder if do not exists
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Get all gt files
    list_gt = os.listdir(gt_folder)

    # Check gt files exists
    if len(list_gt) == 0:
        raise RuntimeError("Error: GT folder is empty. Please, generate the gt files before continue.")

    # Print options #todo: set verbose options
    print("\nFormat dataset {}".format(output_folder))
    print("----------------------------------------")
    print(" Dataset folder: {}".format(dataset_folder))
    print(" Gt Folder: {}".format(gt_folder))
    print(" Orientation: {}".format( orientation_str if orientation else "-"))
    print(" Center Image: {}".format("True" if center else "False"))
    print(" Crop Breast: {}".format("True" if crop_breast else "False"))
    print(" Bias Field N4 {}".format("True" if bias_field_n4 else "False"))
    print(" Save metadata {}".format("True" if save_metadata else "False"))
    print(" Series: {}".format(serie_list if type(serie_list) is list else "All"))
    if type(post_subset) is list:
        print(" Post subset: {}".format(post_subset))
    print(" Overwrite: {}".format("True" if force_rewrite else "False"))
    print("----------------------------------------")

    # TODO: transform metadata to fit with the transformations
    if save_metadata and (orientation is not None or center or crop_breast or bias_field_n4):
        print("Warning: Notice that the saved metadata are extracted from the Dicom Series and do not account for the transformations")

    errors = 0
    # For each folder in the dataset
    for ref in os.listdir(dataset_folder):

        folder = os.path.join(dataset_folder, ref)

        # Get dataset, serie and patient from folder name
        dataset = ref.split("-")[0]
        serie = ref.split("-")[1]
        patient = ref.split("-")[2]

        # filter series
        if type(serie_list) is list and serie not in serie_list:
            continue

        # define output folder
        data_folder = os.path.join(output_folder, ref)

        # Check if already saved (check if gt exists in the OUTPUT folder)
        # Notice: the gt is the last thing saved in the output so, if found, the data has been completely saved before
        gt_new_filename = os.path.join(data_folder, "gt.nii.gz")
        if not force_rewrite and os.path.exists(gt_new_filename):
            if not save_metadata or len([0 for n in os.listdir(data_folder) if n.endswith(".json")]) > 0:
                continue # If not force rewrite and metadata parameters fit, jump this case


        # Start to process this folder

        # Visualization
        print(ref + "...")

        # Check if a GT exists for this case (in gt INPUT folder: gt_folder), and get it
        # Notice: GT can be generated for different template sizes, only the correct one is expected
        #         During the GT generation process a brute force checking is applied to determine the right size
        #         Check the GT generation process for more info
        gt_candidates = [i for i in list_gt if i.startswith(ref)]
        if len(gt_candidates) != 1: # Show error if none or more than one candidate are found
            print("ERROR: GT not found!")
            continue
        else:
            gt = gt_candidates[0]

        # Set ground truth full file name
        gt_filename = os.path.join(gt_folder, gt)

        # Get the acquisition folder date from the GT filename
        # Notice: During the GT generation only a single valid acquisition folder is set
        #         This folder names is saved in the gt file name
        #         In this acquisition folder 1 or more images can fit the GT size and therefore be correct
        acq_folder_date = gt.split("_")[2]

        # Get all acquisition folders in this case
        list_acq = os.listdir(folder)

        # Find the correct acquisition folder that fits the one defined in the GT filename
        acq_candidates = [i for i in list_acq if i.startswith(acq_folder_date)]
        if len(acq_candidates) != 1: # Show error if none or more than one candidate are found
            print("ERROR: Acquisition folder not found!")
            continue
        else:
            acq = acq_candidates[0]


        # Extract information from the acquisition folder
        # Notice: currently not used
        folder_date = acq[0:10]
        folder_time = acq[-5:]
        folder_info = ""
        if len(acq) > 6 + 10:
            folder_info = acq[11:-6]

        # Set acquisition folder full path
        adq_folder = os.path.join(folder, acq)

        # List all images modalities inside the acquisition folder
        list_modalities = os.listdir(adq_folder)

        # Find Pre and Post candidates
        if serie == "E2": #For E2 serie
            pre_candidates = [i for i in list_modalities if "SAG 3D PRE" in i]
            post_candidates = [i for i in list_modalities if "SAG 3D POST" in i]

        elif serie == "BH": #Fore BH serie
            pre_candidates = post_candidates = [i for i in list_modalities if
                                                "-Ax Vibrant MULTIPHASE-" in i]  # A single file contains both
        else:
            raise RuntimeError("Unsupported series")


        # DEPRECATED
        # Set number of components manually
        # Currently only used for testing the LoadDicomSerie new automatic num. components function
        num_components_pre = num_components_post = 1
        if serie == "BH":
            if acq == "10-18-1999-BREAST MRI-60819" or acq == "12-07-1999-BREAST MRI-15168":  # special cases TCGA-BH-A0B6, TCGA-BH-A0E9
                num_components_pre = num_components_post = 7
            else:
                num_components_pre = num_components_post = 5
        elif serie == "E2":
            num_components_post = 3
        else:
            raise RuntimeError("Unsupported series")


        # Get the GT file image size
        try:
            gt_size = GetSize(gt_filename)
        except Exception as e: #Seems there is a problem with the gt file...
            print("Exception: {}\nHint: Try to re-run it with the force_rewrite option <True>".format(e))


        # Filter the pre candidates folders with correct size (using the gt size)
        pre = []
        for c in pre_candidates:
            dicom_pre = os.path.join(adq_folder, c)
            try:
                pre_size = GetSize(dicom_pre)  # , num_components_pre)
                if pre_size == gt_size:
                    pre.append(dicom_pre)
            except Exception as e:
                print("Exception: {}".format(e))

        # Filter the post candidates folders with correct size (using the gt size)
        post = []
        for c in post_candidates:
            dicom_post = os.path.join(adq_folder, c)
            try:
                post_size = GetSize(dicom_post)  # , num_components_post)
                if post_size == gt_size:
                    post.append(dicom_post)
            except Exception as e:
                print("Exception: {}".format(e))

        # Check if at least 1 candidate is found, if not error!
        if len(pre) == 0 or len(post) == 0:
            print("ERROR: No pre or post found. Pre: {}-{} Post: {}-{}".format(len(pre_candidates), len(pre), len(post_candidates), len(post)))
            continue

        # Check if number of candidates bigger than 1
        elif len(pre) > 1 or len(post) > 1:

            # If more than one file is found, filter candidates based on an exception list
            if patient in exceptions_list:
                pre_name, post_name = exceptions_list[patient]

                pre_dicom_folder = None
                for pre_item in pre:
                    if pre_item.endswith(pre_name):
                        pre_dicom_folder = pre_item

                post_dicom_folder = None
                for post_item in post:
                    if post_item.endswith(post_name):
                        post_dicom_folder = post_item

                # IF the exception in not found, error
                if pre_dicom_folder is None or post_dicom_folder is None:
                    raise RuntimeError("The defined exception has not been found!")

            else: #if the case is not expected
                if discard_unknown_exceptions:
                    print("Unknown exception discarded...")
                    continue
                else:
                    print("INFO:", pre, post)
                    raise RuntimeError("Unexpected number of folders.")

        else:
            pre_dicom_folder = pre[0]
            post_dicom_folder = post[0]


        # Load Dicom files

        if serie == "BH":
            # Load files
            metadata = [] if save_metadata else None
            images, num_components = LoadDicomSeries(pre_dicom_folder, metadata, True, 0, -2, verbose=2)

            # Check data
            if len(images) < 2 or (save_metadata and len(images) != len(metadata)):
                raise RuntimeError("Unexpected error loading dicom file {}".format(pre_dicom_folder))

            pre_images = images[:1]
            post_images = images[1:]
            if save_metadata:
                pre_metadata = metadata[:1]
                post_metadata = metadata[1:]

            # TESTING ONLY: Check num_components are correct
            if len(pre_images) != 1 or len(post_images) != num_components_post - 1:
                raise RuntimeError("Unexpected behaviour of num_components 001")

        elif serie == "E2":

            # Load files
            pre_metadata = [] if save_metadata else None
            pre_images, num_components = LoadDicomSeries(pre_dicom_folder, pre_metadata, True, 0, -2, verbose=2)

            post_metadata = [] if save_metadata else None
            post_images, num_components = LoadDicomSeries(post_dicom_folder, post_metadata, True, 0, -2, verbose=2)

            # Check data
            if len(pre_images) != 1 or len(post_images) < 1 or (
                    save_metadata and (len(pre_images) != len(pre_metadata) or len(post_images) != len(post_metadata))):
                print(len(pre_images), len(post_images), len(pre_metadata), len(post_metadata))
                raise RuntimeError("Unexpected error loading dicom file {}".format(pre_dicom_folder))

            # TESTING ONLY: Check num_components are correct
            if len(pre_images) != num_components_pre or len(post_images) != num_components_post:
                raise RuntimeError("Unexpected behaviour of num_components 002")
        else:
            raise RuntimeError("Unsupported series")

        #Filter by post_components list
        if type(post_subset) is list and len(post_subset) > 0:
            post_images = [post_comp for idx, post_comp in enumerate(post_images) if idx+1 in post_subset]

        # Load gt
        gt_image = sitk.ReadImage(gt_filename)

        # Process files
        if crop_breast or center or orientation is not None or bias_field_n4:

            if serie == "BH":
                crop_region = GetBreastCropedRegion(pre_images[0], 50) if crop_breast else None

                #[0, 0, 0, 250, 0]

                gt_num_voxels = CountNumPosPixels(gt_image)
                if gt_num_voxels == 0:
                    print("GT data is empty!")
                    if not save_error_gt_cases:
                        continue
                    else:
                        data_folder = os.path.join(output_folder, "ERROR GTEMPTY " + ref.replace("-"," "))
                        gt_new_filename = os.path.join(data_folder, "gt.nii.gz")

                gt_image = ProcessImage(gt_image, orientation=orientation, force_center=center,crop_region=crop_region)
                if gt_num_voxels != CountNumPosPixels(gt_image):
                    print("GT data has been lost during processing")
                    if not save_error_gt_cases:
                        continue
                    else:
                        data_folder = os.path.join(output_folder, "ERROR GTCROPPED " + ref.replace("-"," "))
                        gt_new_filename = os.path.join(data_folder, "gt.nii.gz")

                pre_images[0] = ProcessImage(pre_images[0],  orientation=orientation, force_center=center,
                                             crop_region=crop_region, bias_field_n4=bias_field_n4)

                for comp_idx in range(len(post_images)):
                    post_images[comp_idx] = ProcessImage(post_images[comp_idx],
                                                         orientation=orientation, force_center=center,
                                                         crop_region=crop_region, bias_field_n4=bias_field_n4)
            elif serie == "E2":
                crop_region = GetBreastCropedRegion(pre_images[0], 50) if crop_breast else None
                #[0, 0, 0, 100, 0, 0]
                gt_num_voxels = CountNumPosPixels(gt_image)
                if gt_num_voxels == 0:
                    print("GT data is empty!")
                    if not save_error_gt_cases:
                        continue
                    else:
                        data_folder = os.path.join(output_folder, "ERROR GTEMPTY " + ref.replace("-"," "))
                        gt_new_filename = os.path.join(data_folder, "gt.nii.gz")

                gt_image = ProcessImage(gt_image, orientation=orientation, force_center=center, crop_region=crop_region)
                if gt_num_voxels != CountNumPosPixels(gt_image):
                    print("GT data has been lost during processing")
                    if not save_error_gt_cases:
                        continue
                    else:
                        data_folder = os.path.join(output_folder, "ERROR GTCROPPED " + ref.replace("-"," "))
                        gt_new_filename = os.path.join(data_folder, "gt.nii.gz")

                pre_images[0] = ProcessImage(pre_images[0], orientation=orientation, force_center=center,
                                             crop_region=crop_region, bias_field_n4=bias_field_n4)
                for comp_idx in range(len(post_images)):
                    post_images[comp_idx] = ProcessImage(post_images[comp_idx],
                                                         orientation=orientation, force_center=center,
                                                         crop_region=crop_region, bias_field_n4=bias_field_n4)
            else:
                raise RuntimeError("Unsupported series")

        # Save files

        # Create if not exists
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)

        # Pre
        pre_filename = os.path.join(data_folder, "{}_{}_PRE_{}".format(acq, os.path.basename(pre_dicom_folder), 1))
        sitk.WriteImage(pre_images[0], pre_filename + ".nii.gz")
        if save_metadata:
            with open(pre_filename + ".json", 'w') as fp:
                json.dump(pre_metadata[0], fp)

        # Post
        for comp_idx in range(len(post_images)):
            if type(post_subset) is list and len(post_subset) > 0: #Set name base on the component
                post_filename = os.path.join(data_folder,  "{}_{}_POST_{}".format(acq, os.path.basename(post_dicom_folder), post_subset[comp_idx]))
            else:
                post_filename = os.path.join(data_folder,
                                             "{}_{}_POST_{}".format(acq, os.path.basename(post_dicom_folder), comp_idx + 1))
            sitk.WriteImage(post_images[comp_idx], post_filename + ".nii.gz")
            if save_metadata:
                with open(post_filename + ".json", 'w') as fp:
                    json.dump(post_metadata[comp_idx], fp)

        # GT
        # GT is last one to be saved, and is checked when overwriting is not turn to know if a ref has been properly saved
        sitk.WriteImage(gt_image, gt_new_filename)

    print("----------------------------------------")
    print("Finish")



def CountNumPosPixels(image):
    return np.sum(sitk.GetArrayFromImage(image) > 0)


# Generate all possible ground truths files in nifti format (prints all compatible folders by brute force)
def CreateGTs(dicom_folder_parent, les_folder, gt_output_folder, serie_list=["AO", "AR", "BH", "E2"],
              force_rewrite=False):
    # TODO: ADD verbose
    print("\nCreating GT  {}".format(gt_output_folder))
    print("----------------------------------------")
    print(" Dataset folder: {}".format(dicom_folder_parent))
    print(" .les folder: {}".format(les_folder))
    print(" Series: {}".format(serie_list if type(serie_list) is list else "All"))
    print(" Overwrite: {}".format("True" if force_rewrite else "False"))
    print("----------------------------------------")

    discard_wrong = True

    if not os.path.exists(gt_output_folder):
        os.mkdir(gt_output_folder)

    if not force_rewrite:
        gt_exist_list = [n[:12] for n in os.listdir(gt_output_folder) if len(n) > 12 and n.endswith(".nii.gz")]

    errors = 0
    for file in os.listdir(les_folder):  # For all les files in les_folder
        if file.endswith(".les"):
            les_filename = os.path.join(les_folder, file)
            full_ref = ref = os.path.splitext(file)[0]

            # Find reference name, sequence and lesion number
            lesion_num = 1
            if full_ref[-2:-1] == "-":
                ref = full_ref[:-2]
                lesion_num = int(full_ref[-1])

            sequence = 1
            if ref[-3:-1] == "-S":
                sequence = int(ref[-1])
                ref = ref[:-3]

            serie = ref.split("-")[1]
            patient = ref.split("-")[2]

            # Filter by serie
            if type(serie_list) is list and not serie in serie_list:
                continue

            # check that the file do not exits
            if not force_rewrite and ref in gt_exist_list:
                continue

            print("REF:" + ref + ", " + str(sequence) + ", " + str(lesion_num))

            # Find if main folder exists
            data_folder = os.path.join(dicom_folder_parent, ref)
            if not os.path.exists(data_folder):
                print("Fail. Folder %s do not exists!" % (data_folder))
                continue

            num = 0

            # List of masks
            list_masks_dirs = []
            list_masks_sizes = []
            list_masks_names = []

            # List adquisition folders
            for adq in os.listdir(data_folder):

                # Extract info
                folder_date = adq[0:10]
                folder_time = adq[-5:]
                folder_info = ""
                if len(adq) > 6 + 10:
                    folder_info = adq[11:-6]

                # print(" > Folder date: %s   Time: %s  Info: %s" % (folder_date, folder_time, folder_info) )

                data_adq_folder = os.path.join(data_folder, adq)

                # Find inside the folder
                for subfolder in os.listdir(data_adq_folder):

                    id = int(subfolder.split("-")[0])
                    id_time = int(subfolder.split("-")[-1])
                    name_tag = subfolder.split("-")[1]

                    dicom_folder = os.path.join(data_adq_folder, subfolder)
                    output_file = os.path.join(gt_output_folder,
                                               full_ref + "_seg_" + folder_date + "_" + str(id) + "-" + str(
                                                   id_time) + ".nii.gz")
                    # output_file = os.path.join(output_folder, full_ref + "_seg_" + folder_date + ".nii.gz")

                    # REF:TCGA - BH - A0B6
                    # Discart wrong cases
                    if discard_wrong:
                        if serie == "AR" and (
                                adq == "05-07-2003-MRI BREAST BILATERAL-81050" or adq == "10-18-2001-MRI - BREAST-52081"):  # TCGA-AR-A1AQ and TCGA-AR-A1AN
                            continue
                        elif serie == "E2" and patient == "A1LG" and (
                                id == 3 or id == 13 or id == 10):  # 3 is corrupted, Another PRE and POST with a single image, needed?
                            continue
                        # if serie == "E2" and patient == "A1LG" and id == 3: #Data is corrupted
                        #    continue

                    # Set num_components manually for some cases
                    num_components = 1
                    if serie == "BH" and name_tag == "Ax Vibrant MULTIPHASE":
                        if adq == "10-18-1999-BREAST MRI-60819" or adq == "12-07-1999-BREAST MRI-15168":  # special cases TCGA-BH-A0B6, TCGA-BH-A0E9
                            num_components = 7
                        else:
                            num_components = 5
                    elif serie == "E2" and name_tag == "SAG 3D POST":
                        num_components = 3

                    # Check if already created a mask with this size before
                    try:
                        file_size = GetSize(dicom_folder, num_components)
                    except Exception as e:
                        # print("File cannot be opened properly: {}\nError: {}".format(dicom_folder, e))
                        continue

                    try:
                        idx = list_masks_sizes.index(file_size)
                        list_masks_dirs[idx].append(str(id) + "-" + str(id_time))
                        continue
                    except:
                        pass

                    # Known files for each serie
                    # For AO serie (Two sequence)
                    # 100 Not fat seq 1
                    # 101 Not fat seq 2
                    # 102 Pre seq 1
                    # 103 Pre seq 2
                    # 104, 106, 108, 110?? Post seq 1
                    # 105, 107, 109, 111??? Post seq 2

                    # For AR
                    # 100, 101, 102 ? seq 1 or 2??

                    # For BH
                    # 100, 101, 102...
                    # for BH id 4 seems to be be correct but may require some special tuning as gt seem displaced 1 slide or so... (ex: A0BG-1_seg_06-11-2001_4)

                    # FOR E2 use name label SAG 3D
                    # PRE single 3D image
                    # POST multiple times in one image  (seems ok but have the problem of time, need modifcation)

                    if (serie == "AO" and ((sequence == 1 and id == 100) or (sequence == 2 and id == 101))) \
                            or (serie == "AR" and (id == 100 or id == 101 or id == 102)) \
                            or (serie == "BH" and (name_tag == "Ax Vibrant MULTIPHASE" or id == 100 or id == 101)) \
                            or (serie == "E2" and (name_tag == "SAG 3D PRE" or name_tag == "SAG 3D POST")):
                        # print(" - %s " % (id))

                        if les2nifti(les_filename, dicom_folder, output_file, num_components) == 0:
                            num = num + 1
                            mask_size = GetSize(output_file)

                            list_masks_names.append(output_file)
                            list_masks_sizes.append(mask_size)
                            list_masks_dirs.append([str(id) + "-" + str(id_time)])
                        else:
                            print ("FAIL: " + dicom_folder)
                            errors += 1

            if num == 1:
                for i in range(0, len(list_masks_names)):
                    print(" Seg: " + list_masks_names[i] + " Size: " + str(list_masks_sizes[i]) + " Dirs: " + str(
                        list_masks_dirs[i]))
            else:
                errors += 1

    if errors > 0:
        print("----------------------------------------")
        print("Finish! ERRORS FOUND: {}".format(errors))
    else:
        print("----------------------------------------")
        print("Finish!")




# Convert a .les file to NIFTI by using a DICOM series as a template
def les2nifti(les_file, dicom_folder, output_file, num_comp=1):
    # Check files and folder
    if not os.path.exists(les_file):
        print("ERROR: File %s do not exists!" % (les_file))
        exit(-1)

    if not os.path.exists(dicom_folder):
        print("ERROR: Folder %s do not exists!" % (dicom_folder))
        exit(-1)

    if not os.path.exists(os.path.dirname(output_file)):
        print("ERROR: Folder %s do not exists!" % (os.path.dirname(output_file)))
        exit(-1)

    # print("Input .les file: %s" % (les_file))
    # print("Input dicom folder: %s" % (dicom_folder))

    # print("\nProcessing...\n")

    # Open les file
    with open(les_file, mode='rb') as file:
        data = file.read()

    # Unpack bounding box
    # File format: y_start, x_start, z_start, y_end, x_end, z_end
    bb = struct.unpack("HHHHHH", data[:12])  # H is unsigned short (2 Bytes= 16 bits)  2*6=12

    # Define BB size and total number of elements
    size_bb = (bb[4] - bb[1] + 1, bb[3] - bb[0] + 1, bb[5] - bb[2] + 1)  # x,y,z BB sizes
    num_elem = size_bb[0] * size_bb[1] * size_bb[2]

    # Check size of the file data is correct
    if len(data) != 12 + num_elem:
        print(str(len(data)) + "!=" + str(12 + num_elem))
        raise RuntimeError("File size is inconsistent")

    # Unpack binary mask data
    # Unpack mask in format z, x, y
    mask_bb = np.array(struct.unpack("b" * num_elem, data[12:]), dtype='uint8').reshape(
        [size_bb[2], size_bb[0], size_bb[1]])  # B is int8 1Byte

    # Open dicom image series
    dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_folder)
    org_image = sitk.ReadImage(dicom_names)

    # Get size (x,y,z)
    org_image_size = list(org_image.GetSize())

    org_image_size[
        2] //= num_comp  # Somehow this number can not be extracted easly from dicom series (at least with sitk, seems that itk can based on itk-snap code: https://github.com/pyushkevich/itksnap/blob/7a104c25401ce23599dc8d32ea80f3daae80f379/Logic/ImageWrapper/GuidedNativeImageIO.cxx)

    # NOTE: Seems that when transformed to numpy images are in format z,y,x (inverse of x,y,z)
    mask = np.zeros(org_image_size[::-1], dtype='uint8')  # create array with size z,y,x

    # Check sizes
    if bb[5] >= org_image_size[2] or bb[3] >= org_image_size[1] or bb[4] >= org_image_size[0]:
        print("Dicom image size do not fit the segementation file!")
        return -1

    # Fill the empty array with the bb mask (mask shape is z,y,x and mask_bb is z,x,y)
    mask[bb[2] - 1:bb[5], bb[0] - 1:bb[3], bb[1] - 1:bb[4]] = mask_bb.transpose(0, 2,
                                                                                1)  # Tranpose bb mask from z,x,y to z,y,x

    # create a itk image
    mask_image = sitk.GetImageFromArray(mask, isVector=False)

    # Fill basic
    mask_image.SetSpacing(org_image.GetSpacing())
    mask_image.SetOrigin(org_image.GetOrigin())
    mask_image.SetDirection(org_image.GetDirection())

    # TODO: Solve header problem

    # Write image
    sitk.WriteImage(mask_image, output_file)

    return 0
