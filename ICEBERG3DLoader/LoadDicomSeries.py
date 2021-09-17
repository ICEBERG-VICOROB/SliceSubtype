# LoadDicom function
# Author: Joel Vidal <jolvid@gmail.com>
# Date: 11/03/20
# Version: 0.4
# SimpleITK version: 1.2.4

# Todo: Currently metadata is given as an addition structure.
#       Check if metadata can be integrated in the image as the case of single component
#       Add/Improve comments

import numpy as np
import SimpleITK as sitk
import json
import os

"""
Function to load a Dicom series as a sitk image
This function is intended to solve the shortcomings of the ImageSeriesReader function found in sitk 1.2.4
Notice: Newer SimpleITK versions may have solve these shortcomings

Parameters:
- folder: Dictom series folder
- metadata: None or List where metadata will be saved.   Default: None
            Metadata is saved as a list (channels) of list of dictionaries (slices)
- force_sorting_instance: Force sorting the Dicom series by instance number.  Default: True
                          Useful for cases where sitk fails to load the slices properly
- number_components: Indicates the number of components of the Dicom image.  Default: 0 (auto)
                     If 0, the num. components are auto determined using "Slice Location" and
                     "Image position Patient" tags. If the tags are not found, the num. components
                     is set to 1. (TODO: Rise a runtime error better)
- get_component: Chose a single component to retrieve. Default: -1 (All components)
                 Special param: -1 return a multi-component image
                                -2 returns the components as a list of independent sitk images
- verbose: level of information printed on screen (0: All, 1: Warnings, 2: Errors) Default: 1

Example code:
=================================
metadata = []
images = LoadDicom("./TCGA-BRCA/TCGA-BRCA/TCGA-E2-A15J/06-20-2002-MR BREAST BILATERAL WWO CONT-27666/7-SAG 3D POST-CONTRAST-83725", metadata)

import json
with open("metadata.json", 'w') as fp:
json.dump(metadata, fp)
"""

def LoadDicomSeries(folder, metadata = None, force_sorting_instance = True, number_components = 0, get_component = -1, force_image_param_from_metadata = False, private_metadata = True, verbose = 1):

    if verbose <= 0:
        print("Loading {} dicom series...".format(folder))
        print("- Retrive metadata: {}".format( "True" if type(metadata) is list else "False"))
        print("- Force sorting by #instance: {}".format( "True" if force_sorting_instance else "False"))

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder)
    reader.SetFileNames(dicom_names)
    reader.SetGlobalWarningDisplay(verbose < 2) #warning dispaly related to verbose

    # Activate metadata
    reader.MetaDataDictionaryArrayUpdateOn() #TODO: May activate this if only needed (metada sort by instance)
    reader.SetLoadPrivateTags(private_metadata & (metadata is not None)) #if this is True private metada is loaded

    #If force sorting by instance
    if force_sorting_instance:
        reader.Execute() #Load images & metadata (image not used here)

        #Check if instance number tag is avaliable in the first slide
        if '0020|0013' not in reader.GetMetaDataKeys(0):
            raise RuntimeError("Instance Number tag not found")

        #Sort slides by instance number
        slice_index = []
        for si in range(len(dicom_names)):
            slice_index.append(int(reader.GetMetaData(si, '0020|0013')))  # Header Instance Number

        #Resort dicom files names based on instance number
        dicom_names = [dicom_names[i] for i in np.argsort(slice_index)]  # Resort by Instance number
        reader.SetFileNames(dicom_names)  # reload files in the correct order

        # print(np.argsort(slice_index))
        # print(dicom_names_fixed)

    #Load image & metadata
    image = reader.Execute()

    if verbose <= 0:
        print("- Number of components: {}".format("Auto" if number_components == 0 else number_components))

    if image.GetNumberOfComponentsPerPixel() > 1:
        if number_components != image.GetNumberOfComponentsPerPixel():
            if verbose <= 1:
                print("WARNING: Wrong number of components set. Change to {}. File: {}".format(image.GetNumberOfComponentsPerPixel(), folder))
        number_components = image.GetNumberOfComponentsPerPixel()



    if force_image_param_from_metadata:
        meta_slice = 0 #Todo: Check the paramaters for all slices and force consistency
        # Find metadata: Patien pose  (0020|0032), Image Orientation (0020|0037), pixel spacing (0028|0030),
        # Slice Thinkness (0018,0050) and/or Spacing Between Slices (0018,0088)
        # More: https://dicom.innolitics.com/ciods/mr-image/mr-image/00180088
        if not reader.HasMetaDataKey(meta_slice,"0020|0032") or not reader.HasMetaDataKey(meta_slice,"0020|0037") or not reader.HasMetaDataKey(meta_slice,"0028|0030") or \
                (not reader.HasMetaDataKey(meta_slice,"0018|0050") and not reader.HasMetaDataKey(meta_slice,"0018|0088")):
            raise RuntimeError("Metdata image parameters not found in slice ({}). These parameters cannot be forced in this image!".format(meta_slice))

        # Get parameters
        patient_position = reader.GetMetaData(meta_slice,"0020|0032").split('\\') # For a single \ we must write \\ as a special character
        image_orientation = reader.GetMetaData(meta_slice, "0020|0037").split('\\')
        pixel_spacing = reader.GetMetaData(meta_slice, "0028|0030").split('\\')
        slice_spacing = reader.GetMetaData(meta_slice, "0018|0088") if reader.HasMetaDataKey(meta_slice,"0018|0088") else reader.GetMetaData(meta_slice, "0018|0050") #spacing between lines has priority over slice thickness

        # Check paramters format and trasnform to numbers
        if len(patient_position) != 3 or len(image_orientation) != 6 or len(pixel_spacing) != 2:
            raise RuntimeError("Unexpected Metadata image parameters format!")

        patient_position = tuple(map(float, patient_position))
        image_orientation = list(map(float, image_orientation))
        image_orientation_matrix = np.array([image_orientation[:3], image_orientation[3:6], np.cross(image_orientation[:3], image_orientation[3:6])]) #compute third row by cross product first two rows (ortonomal matrix)
        direction = tuple(image_orientation_matrix.transpose().reshape(-1))
        spacing = tuple(map(float, pixel_spacing)) + (float(slice_spacing),) #last comma is to indicate is a tuple (not a parhentesis)

        if verbose <= 0:
            print("Force metadata image parameters: ")
            print("\t - Origin: {} -> {}".format(image.GetOrigin(), patient_position))
            print("\t - Direction: {} -> {}".format(image.GetDirection(), direction))
            print("\t - Spacing: {} -> {}".format(image.GetSpacing(), spacing))

        image.SetOrigin(patient_position)
        image.SetDirection(direction)
        image.SetSpacing(spacing)


    #May check metadata consitency: EX: num slide and thiknes equal to image width

    # Determine number of components by Slice Location tag (the slices goes from - to +)
    if number_components == 0:

        #Check if slide location is in the first slide
        if '0020|1041' in reader.GetMetaDataKeys(0):
            #TODO: may identify the first slides
            #Deterime number of components using the slide location tage
            number_components = 1
            slide_local = float(reader.GetMetaData(0, '0020|1041'))  #  Slice Location tag
            for si in range(1,len(dicom_names)):
               v = float(reader.GetMetaData(si, '0020|1041'))  #  Slice Location tag
               if slide_local <= v:
                   slide_local = v
               else: #If the value have decrease w.r.t last slide, means that another component has been found
                   slide_local = v
                   number_components += 1

        elif '0020|0032' in reader.GetMetaDataKeys(0): #Image position Patient

            #check if data format seems correct
            if len(reader.GetMetaData(0, '0020|0032').split("\\")) != 3:
                number_components = 1
                if verbose <= 0:
                    print("WARNING: Image Position Patient has unexpected format File: %s. Default number components used: %d" % (folder, number_components))
            else:
                #Deterime number of components using the slide location tage
                number_components = 1
                slide_local = float(reader.GetMetaData(0, '0020|0032').split("\\")[2])  #  Slice Location tag
                for si in range(1,len(dicom_names)):
                   v = float(reader.GetMetaData(si, '0020|0032').split("\\")[2])  #  Slice Location tag
                   if slide_local <= v:
                       slide_local = v
                   else: #If the value have decrease w.r.t last slide, means that another component has been found
                       #TODO: Error, seems that some image has the slide order inverted so all slices are detected as components
                       slide_local = v
                       number_components += 1

        else: #TODO: May raise a runtime error!
            number_components = 1
            if verbose <= 0:
                print("WARNING: Slide location metadata not found. Default number components used: %d" % (number_components))

        if verbose <= 0:
            print("- Auto-determined number of components: {}".format(number_components))

        #if number_components > 1: #If number different than 1, show info
            #    print("[INFO] Number of components found: %d" % (number_components))

    if verbose <= 0:
        print("- Extract components: {}".format("All" if get_component < 0 else get_component))

    if number_components > 1: #Multi-component found, create a 4D image

        # transform to np array
        image_array = np.array(sitk.GetArrayFromImage(image))  # Load as z,y,x


        if len(image_array.shape) == 3: #If the image is not 4D, it needs to be divided by slides

            #Check for unexpected errors
            if image.GetNumberOfComponentsPerPixel() > 1:
                raise RuntimeError("Unexpected error while identifying image components")

            # Check if can be divided properly
            if image_array.shape[0] % number_components != 0:
                print(folder, image_array.shape[0], number_components )
                raise RuntimeError("Wrong number of components")

            # Compute number of slides per component
            n_slices = image_array.shape[0] // number_components

            # Create a 4D image
            vector_image = np.zeros((n_slices, image_array.shape[1], image_array.shape[2], number_components),
                                    dtype=image_array.dtype)

            # Fill each new dimension with components
            for i in range(0, number_components):
                vector_image[:, :, :, i] = image_array[i * n_slices:(i + 1) * n_slices, :, :]

        elif len(image_array.shape) == 4 and image_array.shape[3] == number_components:
            vector_image = image_array
        else:
            raise RuntimeError("Unexpected image format")

        # Create sitkImage
        if get_component >= 0 and get_component < number_components:  #only single components
            vector_image_sitk = sitk.GetImageFromArray(vector_image[:, :, :, get_component])
            vector_image_sitk.SetSpacing(image.GetSpacing())
            vector_image_sitk.SetDirection(image.GetDirection())
            vector_image_sitk.SetOrigin(image.GetOrigin())
            image = vector_image_sitk
        elif get_component == -1:  #All components
            vector_image_sitk = sitk.GetImageFromArray(vector_image)
            vector_image_sitk.SetSpacing(image.GetSpacing())
            vector_image_sitk.SetDirection(image.GetDirection())
            vector_image_sitk.SetOrigin(image.GetOrigin())
            image = vector_image_sitk
        elif get_component == -2: #A list with all components
            list_images = []
            for i in range(number_components):
                vector_image_sitk = sitk.GetImageFromArray(vector_image[:, :, :, i])
                vector_image_sitk.SetSpacing(image.GetSpacing())
                vector_image_sitk.SetDirection(image.GetDirection())
                vector_image_sitk.SetOrigin(image.GetOrigin())
                list_images.append(vector_image_sitk)
            image = list_images
        else:
            raise RuntimeError("get_component value out of range!")

    elif get_component == -2: #A list with all components (in this case only one)
            image = [image]

    elif get_component != -1 and get_component != 0: #Different than all components or the first
            raise RuntimeError("get_component value out of range!")

    #If metadata set, save all metadata as list of dictionaries
    if type(metadata) is list:
        del metadata[:] #python3.3 metadata.clear()

        for slideIdx in range(len(dicom_names)):
            slide_dict = {}
            for key in reader.GetMetaDataKeys(slideIdx):
                if reader.HasMetaDataKey(slideIdx, key):
                    slide_dict[key] = reader.GetMetaData(slideIdx, key)

            if number_components > 1: #Multi-component
                comp_idx = int(slideIdx / n_slices)
                if get_component < 0: #All components
                    if len(metadata) <= comp_idx:
                        metadata.append([slide_dict])
                    else:
                        metadata[comp_idx].append(slide_dict)

                elif get_component == comp_idx: #Only single component
                    if len(metadata) == 0:
                        metadata.append([slide_dict])
                    else:
                        metadata[0].append(slide_dict)
            else:
                if len(metadata) == 0:
                    metadata.append([slide_dict])
                else:
                    metadata[0].append(slide_dict)


    elif metadata is not None: #If wrong parameter
        raise RuntimeError("Metadata parameter must be List or None")

    #return sitk imagedicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folder)
    #return sitk.ReadImage(dicom_names)
    #NOTE: May remove  num_components from return??
    return image, number_components



# small metadata name dictionary
try:
    from . dicom_dict import * #Metadata name dictionry
    import re

    #Get Metadata tag Name
    def GetDicomMetadataName(key, space = True): #NEED TO USE upper()
        if space:
            return re.sub(r"(\w)([A-Z])", r"\1 \2", dicom_dict[key.upper()])
        else:
            return dicom_dict[key.upper()]

    def AddMetadataItems(items_dict):
        dicom_dict.update(items_dict)

    #Print Metdata as a table (single component only)
    def PrintMetadata(metadata, subset = "", space = 30, comp = 1): #Todo:space =0 auto len using key len etc..

        if type(metadata) is dict:
            data = [[metadata]]
        elif type(metadata[0]) is not list:
            data = [metadata]
        else:
            data = metadata

        if len(data) < comp:
            raise RuntimeError("Component out of range")

        keys_list = []
        value_list = []
        for m in data[comp - 1]:
            value = []
            for k, v in m.items():
                if subset != "" and not k.startswith(subset):
                        continue
                if len(keys_list) == len(value):
                    if k.upper() in dicom_dict:
                        keys_list.append(dicom_dict[k.upper()][:space-1])
                    else: #if no name is found
                        keys_list.append("({})".format(k.upper()))
                value.append("{}".format(v)[:space-1])
            value_list.append(value)

        #row_format = ("{:>%d}" % (space)) * (len(keys_list) + 1)  #{:>30}
       # if subset != "":
            #print("Subset [{}]".format(subset))

        if len(value_list) == 0:
             print("No data found!")
        else:
            row_format = "{:>5}" + ("{:>%d}" % (space)) * len(keys_list)  # {:>30}
            elems = ["{}".format(i) for i in range(len(value_list))]
            print(row_format.format("Slide", *keys_list))
            for team, row in zip(elems, value_list):
                print(row_format.format(team, *row))


except ImportError:
    print("DICOM Dict not found")


