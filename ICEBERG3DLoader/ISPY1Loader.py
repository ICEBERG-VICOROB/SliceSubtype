from . LoadDicomSeries import *
from . MRIOrientation import TransformMRIOrientation, GetAnatomicalOrientationFromOrientation, GetWorldCoordinatesPos
from . BreastCrop import GetBreastCropedRegion
import os
import itertools
import SimpleITK as sitk
import random
import struct
import json
import sys


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
def AutoFormatDataset(dataset_folder, output_folder, crop_breast=False, orientation=None, center=False,
                      bias_field_n4=False, save_metadata=True, time_list=["t1", "t2", "t3", "t4"], post_subset=None,
                      force_rewrite=True, dataset_name=None):

    # Check input folder
    if not os.path.exists(dataset_folder):
        raise RuntimeError("Input dataset folder do not exists!")

    # Create output folder
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if type(dataset_name) is not str:
        dataset_name = GetDatasetName(crop_breast, orientation, center, bias_field_n4, post_subset)

    formated_folder = os.path.join(output_folder, dataset_name)

    # Create Formated dataset
    CreateFormatedDataset(dataset_folder, formated_folder, crop_breast, orientation, center, bias_field_n4,
                          save_metadata, time_list, post_subset, force_rewrite)

    print("\n> Output dataset folder: {}".format(formated_folder))

    #Load dataset
    dataset = LoadFromattedDataset(formated_folder, time_list)

    return dataset

    #return formated_folder


# Get the Pre, Post/s and Labels from dataset
def FormatFilesLists( dataset, post_idx = 0, shuffle = True, maximum_files=None, time_item = 1 ):

    print("Preparing list...")
    image_pre_list = []
    image_post_list = []
    label_list = []
    name_list = []
    for key, item in dataset.items():
        # print(item)
        # Filter by serie
        if isinstance(time_item, int) and time_item > 0:
            time_item = "t{}".format(time_item)

        if "gt" not in item:
            if time_item in ["t1","t2","t3","t4"]:
                if time_item not in item:
                    print("- Time {} not found for {}".format(time_item, key))
                    continue
                item = item[time_item]
            else:
                for t in ["t1","t2","t3","t4"]:
                    if t in item:
                        item = item[t]
                        break


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



# LOAD AND FORMAT DATASET -----

# Load all filenames of a formatted dataset in a dictonary
# - formated_dataset_folder: folder where the formatted dataset is saved
# - time_list: list of time to be loaded (if found in dataset)
# - single_time: if true only one time will be loaded (if more than one, only the first)
def LoadFromattedDataset(formatted_dataset_folder, time_list=["t1", "t2", "t3", "t4"], single_time = False):
    result = {}

    if time_list is None:
        time_list = ["t1", "t2", "t3", "t4"]

    for ref in os.listdir(formatted_dataset_folder):

        folder = os.path.join(formatted_dataset_folder, ref)

        if len(ref.split("_")) < 2:
            print("Folder {} unknown".format(ref))
            continue


        dataset = ref.split("_")[0]
        patient = ref.split("_")[1]

        list_nifti = [i for i in os.listdir(folder) if i.endswith(".nii.gz")]

        result[ref] = {}
        for time_item in time_list:

            gt_filename = os.path.join(folder, "{}_gt.nii.gz".format(time_item))
            if not os.path.exists(gt_filename):
                continue

            pre_filenames = [os.path.join(folder, file) for file in list_nifti if file.startswith("{}_".format(time_item)) and file.endswith("PRE_1.nii.gz")]
            if len(pre_filenames) != 1:
                continue

            post_filenames = [os.path.join(folder, file)  for file in list_nifti if file.startswith("{}_".format(time_item)) and "_POST_" in file]
            if len(post_filenames) < 2: #only 2 post for this dataset?
                continue

            #CHECK and SORT POST
            post_filenames_sorted = []
            for i in range(1, len(post_filenames)+1):
                candidates = [ p for p in post_filenames if "_POST_{}.nii.gz".format(i) in p ]
                if len(candidates) != 1:
                    raise RuntimeError("Unexpected number of candidates for post case")
                post_filenames_sorted.append(candidates[0])


            if single_time:
                result[ref] = {
                    "gt": gt_filename,
                    "pre": pre_filenames,
                    "post": post_filenames_sorted,
                    "time": time_item,
                    "patient": patient
                }
                break
            else:
                result[ref][time_item]= {
                            "gt": gt_filename,
                            "pre":  pre_filenames,
                            "post": post_filenames_sorted,
                            "serie": time_item,
                            "patient": patient
                        }

        if len(result[ref]) == 0:
            raise RuntimeError("Result missing! {}".format(ref))

    return result



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




def GetSizeAndComponents(file):
    if not os.path.exists(file):
        print("ERROR: %s do not exists!" % (file))
        exit(-1)

    if os.path.isdir(file):
        org_image, num_components = LoadDicomSeries(file, None, True, verbose=2)  # num_component
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

    return tuple(org_image_size), num_components


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


def CountNumPosPixels(image):
    return np.sum(sitk.GetArrayFromImage(image) > 0)


# https://wiki.cancerimagingarchive.net/display/Public/I+SPY-1+DCE+MRI+Data+Sharing+DICOM+Dictionary
"""
----------------------------------------------------------------------------------------------------------
VOILPS              Patient coordinate system specified rectangular VOI Sequence	SQ	(0117,1020)
----------------------------------------------------------------------------------------------------------
> VOILPS            Center	Center of the VOI	                                    DS (3)	(0117,1042)
> VOILPS HalfWidth	1st half dimension vector of the VOI                            DS (3)	(0117,1043)
> VOILPS HalfHeight	2nd half dimension vector of the VOI	                        DS (3)	(0117,1044)
> VOILPS HalfDepth	3rd half dimension vector of the VOI	                        DS (3)	(0117,1045)
> VOILPS Type	    "Use for the specified region. Defined terms:	                CS	    (0117,1046)
                    ""VOI"": Region to be analyzed 
                    ""OMIT"": Region to be excluded from the analysis"
----------------------------------------------------------------------------------------------------------
VOI_pixel_start     (x,y,z) coordinates of the first voxel in the VOI	            US (3)	(0117,10A1)
----------------------------------------------------------------------------------------------------------
VOI_pixel_end       (x,y,z) coordinates of the last voxel in the VOI	            US (3)	(0117,10A2)
----------------------------------------------------------------------------------------------------------
""" #TODO: May add items to dicom_dict?
def VOIFromMetadata(metadata):
    #VOILPS = metadata["0117|1020"]
    #VOILPS_Center = metadata["0117|1042"]
    #VOILPS_HalfWidth = metadata["0117|1043"]
    #VOILPS_HalfHeight = metadata["0117|1044"]
    #VOILPS_HalfDepth = metadata["0117|1045"]
    #VOILPS_Type = metadata["0117|1046"]
    if "0117|10A1" in metadata and "0117|10A2" in metadata:
        VOI_pixel_start = metadata["0117|10A1"]
        VOI_pixel_end = metadata["0117|10A2"]
        return np.array(VOI_pixel_start.split("\\"), dtype=int), np.array(VOI_pixel_end.split("\\"), dtype=int)
    else:
        return None, None

#Return metadata key upcase!
def LoadDicom(image_file):
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_file)
    reader.SetLoadPrivateTags(True)
    image = reader.Execute()
    metadata_dict = {}
    for key in reader.GetMetaDataKeys():
        metadata_dict[key.upper()] = reader.GetMetaData(key)
    return image, metadata_dict


class Filter(object):
    def __init__(self, stream, re_pattern):
        self.stream = stream
        self.pattern = re.compile(re_pattern) if isinstance(re_pattern, str) else re_pattern
        self.triggered = False

    def __getattr__(self, attr_name):
        return getattr(self.stream, attr_name)

    def write(self, data):
        if data == '\n' and self.triggered:
            self.triggered = False
        else:
            if self.pattern.search(data) is None:
                self.stream.write(data)
                self.stream.flush()
            else:
                # caught bad pattern
                self.triggered = True

    def flush(self):
        self.stream.flush()


# Create a formated dataset with preprocessed data and GT files in NIFTI format (NIFTI GTs are required)
def CreateFormatedDataset(dataset_folder, output_folder, crop_breast=False, orientation=None, center=False,
                          bias_field_n4=False, save_metadata=True, time_list=["t1", "t2", "t3", "t4"], post_subset=None,
                          force_rewrite=False):
    # TODO: ADD verbose
    # TODO: Allow to save all components in a single file
    # TODO: Save parameters in a dicctionari and save the file in json to identify the dataset( Load the dataset could be as easy as load the json )

    #Reddirect errors to nuul
    #sys.stdout = Filter(sys.stdout, r'WARNING:')

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

    # Print options #todo: set verbose options
    print("\nFormat dataset {}".format(output_folder))
    print("----------------------------------------")
    print(" Dataset folder: {}".format(dataset_folder))
    print(" Orientation: {}".format( orientation_str if orientation else "-"))
    print(" Center Image: {}".format("True" if center else "False"))
    print(" Crop Breast: {}".format("True" if crop_breast else "False"))
    print(" Bias Field N4 {}".format("True" if bias_field_n4 else "False"))
    print(" Save metadata {}".format("True" if save_metadata else "False"))
    print(" Times: {}".format(time_list if type(time_list) is list else "All"))
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
        dataset = ref.split("_")[0]
        patient = ref.split("_")[1]

        # define output folder
        data_folder = os.path.join(output_folder, ref)

        # Start to process this folder

        # Visualization
        print(ref + "...")

        # Get all acquisition folders in this case
        list_acq = os.listdir(folder)

        # Find the correct acquisition folder that fits the one defined in the GT filename


        acq_items = { "-".join([acq.split("-")[2], acq.split("-")[0], acq.split("-")[1], acq.split("-")[3]]): acq for acq in list_acq}
        if len(acq_items) == 0: #!= 4:
            print("ERROR: Acquisition folders missing! {}".format(folder))
            continue

        #sort
        acq_items = { "t{}".format(i+1): acq_items[key] for i,key in enumerate(sorted(acq_items.keys())) }
        #for k, acq in acq_items.items():
            #print("{}: {}   []".format(k, acq, k in time_list))


        if isinstance(time_list, list) and len(time_list) > 0:
            acq_items = {k: v for k, v in acq_items.items() if k in time_list}

        if len(acq_items) == 0:
            print("ERROR: All adquisitions rejectet!")
            continue

        for time_item, acq in acq_items.items():

            #print("{} Process {}".format(time_item, acq))
            if len(acq) < 18:
                print("ERROR: Folder name error!")
                continue

            folder_date = acq[0:10]
            folder_time = acq[11:17]
            folder_info = acq[18:]

            # Set acquisition folder full path
            acq_folder = os.path.join(folder, acq)

            gt_new_filename = os.path.join(data_folder, "{}_gt.nii.gz".format(time_item))
            if not force_rewrite and os.path.exists(gt_new_filename):
                if not save_metadata or len([0 for n in os.listdir(data_folder) if n.endswith(".json")]) > 0:
                    continue  # If not force rewrite and metadata parameters fit, jump this case

            # List all images modalities inside the acquisition folder
            list_modalities = os.listdir(acq_folder)

            modality = "Dynamic"
            if modality == "Dynamic":
                sub_modelities = [ mod for mod in list_modalities if "-Dynamic-3dfgre-" in mod]
            else:
                print("ERROR: Unknown modality!")
                continue

            #IF more than one pick the one with biggern number of components
            if len(sub_modelities) > 1:
                max_slices = max_comp = 0
                modality = []
                for s in sub_modelities:
                   size, n_comp = GetSizeAndComponents(os.path.join(acq_folder,s))
                   if n_comp > max_comp or n_comp == max_comp and size[2] > max_slices:
                       modality = [s]
                       max_slices = size[2]
                       max_comp = n_comp
                sub_modelities = modality
                print("ERROR: Unexpected number of modialitites [{}] Pick.. {} ({}, {})...!".format(len(sub_modelities), sub_modelities[0], max_slices, max_comp))

            if len(sub_modelities) == 0:
                print("ERROR: No correct modalities found [{}]!".format(len(sub_modelities)))
                continue



            #try to get gt
            gt_modality = "-PE Segmentation thresh70-"
            gt_modelities = [mod for mod in list_modalities if gt_modality in mod]

            if len(gt_modelities) != 1:
                print("ERROR: Unexpected number of gt modalities [{}]!".format(len(gt_modelities)))
                continue

            data = os.path.join(acq_folder, sub_modelities[0])

            gt_folder = os.path.join(acq_folder, gt_modelities[0])
            gt_file_list = [os.path.join(gt_folder, file) for file in os.listdir(gt_folder) ]
            if len(gt_file_list) != 1: #only one file dcm is expected inside the folder
                print("ERROR: GT file not found or the file format is unknown")
                continue
            gt = gt_file_list[0]

            try:
                gt_size = GetSize(gt) #We discart the component number
                data_size = GetSize(data)
            except Exception as e: #Seems there is a problem with the gt file...
                print("Exception: {}\nHint: Try to re-run it with the force_rewrite option <True>".format(e))

            #Check the size of the first the 2 dimensions (gt are croped in the 3rd dimension)
            if gt_size[:2] != data_size[:2]:
                print("ERROR: Nonmatching gt and data sizes.")
                continue

            metadata = []
            images, num_components = LoadDicomSeries(data, metadata, True, 0, -2, True, verbose=2) #Force image parameters from metadata due to problems with spacing

            # Check data
            if len(images) < 2 or len(images) != len(metadata):
                raise RuntimeError("Unexpected error loading dicom file {}".format(data))

            first_slice_metadata = metadata[0][0]
            pre_images = images[:1]
            post_images = images[1:]
            if save_metadata:
                pre_metadata = metadata[:1]
                post_metadata = metadata[1:]

            #Filter by post_components list
            if type(post_subset) is list and len(post_subset) > 0:
                post_images = [post_comp for idx, post_comp in enumerate(post_images) if idx+1 in post_subset]

            # Load gt
            gt_image, gt_metadata = LoadDicom(gt)#sitk.ReadImage(gt)



            #Get VOI from first slice
            VOI = VOIFromMetadata(gt_metadata)
            if VOI[0] is not None:
                print("VOID Found ({})".format(VOI))
                gt_square = np.zeros(data_size[::-1])
                gt_square[VOI[0][2]:VOI[1][2], VOI[0][1]:VOI[1][1], VOI[0][0]:VOI[1][0]] = 1
                gt_square = sitk.GetImageFromArray(gt_square)
                gt_square.SetOrigin(images[0].GetOrigin())
                gt_square.SetDirection(images[0].GetDirection())
                gt_square.SetSpacing(images[0].GetSpacing())
                gt_image = gt_square

            # Process gt images:
            # GT image segmentation are cropped in the 3rd dimension.
            # Here we use metadata (origin, orientation and spacing) to locate the relative position of the gt
            # and resize it to the same image than the MRI by padding

            # find the axis of the 3rd anatomical direction
            direction_matrix = np.array(images[0].GetDirection()).reshape(3, 3)
            axis = np.where(direction_matrix[:, 2])[0][0]
            # print(np.where(direction_matrix[:,0]),np.where(direction_matrix[:,1]),np.where(direction_matrix[:,2])) #axis of anatomical adirection

            # Compute offset in number of slices (3rd image's dimension) between GT's Origin and Image's Origin
            # The origin of the images is given by the position with respect the patient. Then we find the
            # anatomical axis that correpsonts to the 3rd image dimension using the direction matrix.
            # To find the offset in number of slices, the offset in mm is divided by the slice seperation (also in mm)
            origin_slices_offset = int((images[0].GetOrigin()[axis] - gt_image.GetOrigin()[axis]) / images[0].GetSpacing()[2] + 0.5) #0.5 is to round
            end_slice_difference = images[0].GetSize()[2] - gt_image.GetSize()[2] - origin_slices_offset

            # Pad with zero the res of the gt image
            gt_image = sitk.ConstantPad(gt_image, [0,0,origin_slices_offset], [0,0,end_slice_difference])

            #Threshold the image to binnari(0-1) (everything different than 0)
            # https://simpleitk.org/doxygen/latest/html/namespaceitk_1_1simple.html#a4bb33a74c6a6f8a7e04cd135251c4e55
            gt_image = sitk.NotEqual(gt_image, 0)

            # OTHER not used
            #print("GT ", gt_image.GetSize(), gt_image.GetSpacing(), gt_image.GetOrigin())
            #print("Images: ", images[0].GetSize(), images[0].GetSpacing(), images[0].GetOrigin())
            # compute real-world pose (not needed)
            # start_gt = GetWorldCoordinatesPos([0,0,0], gt_image.GetDirection(),gt_image.GetSpacing(), gt_image.GetOrigin())
            # end_gt = GetWorldCoordinatesPos(np.array(gt_image.GetSize())-np.array([1,1,1]), gt_image.GetDirection(),gt_image.GetSpacing(), gt_image.GetOrigin())
            # start_im = GetWorldCoordinatesPos([0,0,0], images[0].GetDirection(),images[0].GetSpacing(), images[0].GetOrigin())
            # end_im = GetWorldCoordinatesPos(np.array(images[0].GetSize())-np.array([1,1,1]), images[0].GetDirection(),images[0].GetSpacing(), images[0].GetOrigin())
            #print(start_gt, end_gt)
            #print(start_im, end_im)





            # Process files
            if crop_breast or center or orientation is not None or bias_field_n4:

                crop_region = GetBreastCropedRegion(pre_images[0], 50) if crop_breast else None

                #[0, 0, 0, 250, 0]

                gt_num_voxels = CountNumPosPixels(gt_image)
                if gt_num_voxels == 0:
                    print("GT data is empty!")
                    if not save_error_gt_cases:
                        continue
                    else:
                        data_folder = os.path.join(output_folder, "ERROR GTEMPTY " + ref.replace("-"," "))
                        gt_new_filename = os.path.join(data_folder, "{}_gt.nii.gz".format(time_item))

                gt_image = ProcessImage(gt_image, orientation=orientation, force_center=center,crop_region=crop_region)
                if gt_num_voxels != CountNumPosPixels(gt_image):
                    print("GT data has been lost during processing")
                    if not save_error_gt_cases:
                        continue
                    else:
                        data_folder = os.path.join(output_folder, "ERROR GTCROPPED " + ref.replace("-"," "))
                        gt_new_filename = os.path.join(data_folder, "{}_gt.nii.gz".format(time_item))

                pre_images[0] = ProcessImage(pre_images[0],  orientation=orientation, force_center=center,
                                             crop_region=crop_region, bias_field_n4=bias_field_n4)

                for comp_idx in range(len(post_images)):
                    post_images[comp_idx] = ProcessImage(post_images[comp_idx],
                                                         orientation=orientation, force_center=center,
                                                         crop_region=crop_region, bias_field_n4=bias_field_n4)


            # Save files

            # Create if not exists
            if not os.path.exists(data_folder):
                os.mkdir(data_folder)

            # Pre
            pre_filename = os.path.join(data_folder, "{}_{}_PRE_{}".format(time_item, os.path.basename(data), 1))
            sitk.WriteImage(pre_images[0], pre_filename + ".nii.gz")
            if save_metadata:
                with open(pre_filename + ".json", 'w') as fp:
                    json.dump(pre_metadata[0], fp)

            # Post
            for comp_idx in range(len(post_images)):
                if type(post_subset) is list and len(post_subset) > 0: #Set name base on the component
                    post_filename = os.path.join(data_folder,  "{}_{}_POST_{}".format(time_item, os.path.basename(data), post_subset[comp_idx]))
                else:
                    post_filename = os.path.join(data_folder,
                                                 "{}_{}_POST_{}".format(time_item, os.path.basename(data), comp_idx + 1))
                sitk.WriteImage(post_images[comp_idx], post_filename + ".nii.gz")
                if save_metadata:
                    with open(post_filename + ".json", 'w') as fp:
                        json.dump(post_metadata[comp_idx], fp)

            # GT
            # GT is last one to be saved, and is checked when overwriting is not turn to know if a ref has been properly saved
            sitk.WriteImage(gt_image, gt_new_filename)

    print("----------------------------------------")
    print("Finish")

