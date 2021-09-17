# Tools to crop the breast part of the image
# Author: Joel Vidal <jolvid@gmail.com>
# Date: 26/05/20
# Version: 0.2

# Todo: Improve/Simplify the FindSternum2D function
#       Add/Improve comments

from . MRIOrientation import GetCurrentAnatomicalOrientation
import SimpleITK as sitk
import numpy as np


# Crop the breast part of the image
# It uses the image anatomical orientation to identify the sternum position and crop the breast part of the image
# Parameters:
# - image: sitk breast image (with correct anatomical orientation)
# - margin: Distance in voxels between sternum and cutting point (into the breast direction)
# - voxel_threshold: threshold value for a voxel to be considered as a breast part
def CropBreast(image, margin = 0, voxel_treshold=0):

    #Transform image to array
    image_array = sitk.GetArrayFromImage(image)

    #Get sternum position, axis and direction
    pos, axis, dir = FindSternum2D(image, voxel_treshold=voxel_treshold)

    #print(pos, axis, dir)

    # Create crop slice
    crop_slice = [slice(0, s) for s in image_array.shape]
    crop_slice[2-axis] = slice(0, pos + margin) if dir > 0 else slice(pos - margin, image_array.shape[2-axis])

    #crop image
    image_array = image_array[tuple(crop_slice)]

    #backup image info
    spacing, origin, direction = image.GetSpacing(), image.GetOrigin(), image.GetDirection()

    #Create new sitk image
    image = sitk.GetImageFromArray(image_array)

    # Restore info
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)

    return image

# Get the crop roi of the breast for a given image
# The roi cuts the 3D image in half based on the sternum position
# An additional margin w.r.t the sternum can be manually added
# Notice: the code uses the anatomical direction to determine the cutting axis
def GetBreastCropedRegion(image, margin = 0, voxel_threshold=0):

    #Get sternum position, axis and direction
    pos, axis, dir = FindSternum2D(image, voxel_treshold=voxel_threshold)

    #Transform image to array
    image_array = sitk.GetArrayFromImage(image)

    #print(pos, axis, dir)

    # Create crop #todo: adapt to slice tuple?
    crop_region = [0, image_array.shape[2], 0, image_array.shape[1], 0, image_array.shape[0]]
    crop_region[(2-axis)*2:(2-axis)*2+2] = [0, pos + margin] if dir > 0 else [image_array.shape[(2-axis)] - pos - margin, image_array.shape[(2-axis)]]
    #print(crop_region)
    return tuple(crop_region)


# Determine the axis of an Anatomical orientation
def DetermineAxisInAnatomicalOrientaion(orientation, term):
    if term == "R" or term == "L":
        oposite_term = "L" if term == "R" else "R"
    elif term == "A" or term == "P":
        oposite_term = "P" if term == "A" else "A"
    elif term == "S" or term == "I":
        oposite_term = "I" if term == "S" else "S"
    else:
        raise RuntimeError("Unkown term!")

    for i in range(3):
        if orientation[i] == term or orientation[i] == oposite_term:
            return i, 1 if orientation[i] == term else -1

    raise RuntimeError("Term not found!")

# Find the Sternum position, axis and direction
# The code determines de axis and direction based on the anatomical orientation
# Notice that the breast must be centered
def FindSternum2D(image, voxel_treshold=10, min_voxels = 10, middle_region_width=8, middle_region_step=1, horizontal_line_thickness=1, horizontal_line_step=1):

    if len(image.GetSize()) > 3:
        raise RuntimeError("Number of dimensions currently not supported!") #todo: need to adapt and check

    anatomical_orientation = GetCurrentAnatomicalOrientation(image)

    #First find the middle of the patient body (half slide between Anteriori and Posteriori)
    axis_is, dir_is = DetermineAxisInAnatomicalOrientaion(anatomical_orientation, "I")

    image_array = sitk.GetArrayFromImage(image)

    # Get middle slide of the axis (Remmeber: numpy axis are inverse than sitk)
    slide = image_array[ tuple([ image_array.shape[i]//2 if (axis_is == (2-i)) else slice(0, image_array.shape[i]) for i in range(2)]) ]


    if len(slide.shape) == 3:
        raise RuntimeError("Currently not supported!") #todo: need to check this part
        #TODO: Maybe allow sum channels or another option?? to create a single one insted of select a single
        #Select a channel
        if num_channel is not None:
            if 0 <= num_channel < slide.shape[-1]:
                slide = slide[...,num_channel]
            else:
                raise RuntimeError("Num channel out of range")
        else:
            print("WARNING! Unexpected channel dimension found. Using channel 0 by default.")
            slide = slide[..., 0]
    elif len(slide.shape) > 3:
        raise RuntimeError("Unsupported number of dimensions")


    #Determine the direction of the search, from top patient to bottom (supperior to inferior9 (here direction is important)
    if dir_is > 0:
        anatomical_orientation_slide = anatomical_orientation.replace('I','') #Remove the A term, only two positions missing like the dimension in the slide
    else:
        anatomical_orientation_slide = anatomical_orientation.replace('S', '')

    # Determine the A->P axis and its direction
    axis_ap, dir_ap = DetermineAxisInAnatomicalOrientaion(anatomical_orientation_slide, "A")

    #cheking_area_length = 10 #10 pixels
    #cheking_area_step = 1
    #voxel_treshold = 10
    #min_voxels = 10
    if voxel_treshold <= 0: #auto-define the voxel theshold
        voxel_treshold = np.median(slide) / 2

    # How we found the sternum?
    #
    # For the middle slide of the breast
    # We search for the region in a centered region that have values bigger than a threshold
    #
    # Search region:
    #   ------#####------ horizontal_line 0   [A]      _       _
    #   ------#####------ horizontal_line 1    |     /  \    /  \ <-- breast representation
    #   ------#####------ horizontal_line 2    |    {    \__|    \
    #   ------#####------ horizontal_line 3   [P]          ^ sternum pos
    #     middle_region                          [L]--------------[R]
    #
    # We look for the sternum in the search region (#)
    # This zone is defined by two axis:
    # - horizontal_lines: horizontal lines to check (From A-P, all the image) (have thickness and sampling step )*
    # - middle_region: centered area to check in the horizontal axis (A-P) (has width and sampling step)
    #
    # Searching algorithm:
    # For each column we look for the first voxel of the sternum zone.
    # The sternum zone is the first continuous region bigger than min_voxels with values bigger than voxel_threshold.
    # The final sternum point is obtained by the mean of all columns.

    #todo: may simplify this part by removing some currently not used parameter as horizontal_line_thickness

    if axis_ap == 0: # if A->P is axis 0 (Remember: np axis inverse than sitk)

        # Define horizontal lines from A to P
        horizontal_lines = range(0, slide.shape[1], horizontal_line_step) if dir_ap > 0 else range(slide.shape[1] - 1, -1, -horizontal_line_step) #step_size=1
        # Define middle cropping zone of each lines
        middle_region = slice(slide.shape[0] // 2 - middle_region_width // 2, slide.shape[0] // 2 - middle_region_width // 2 + middle_region_width, middle_region_step)

        crop_lines = zip([middle_region] * slide.shape[1], [slice(i - horizontal_line_thickness // 2, i - horizontal_line_thickness // 2 + horizontal_line_thickness) for i in horizontal_lines])

    else: # if A->P is axis 1  (Remember: np axis inverse than sitk)

        #  Define horizontal lines from A to P (-)
        horizontal_lines = range(0, slide.shape[0], horizontal_line_step) if dir_ap > 0 else range(slide.shape[0]-1, -1, -horizontal_line_step)
        # Define middle cropping zone of each lines
        middle_region = slice(slide.shape[1] // 2 - middle_region_width // 2, slide.shape[1] // 2 - middle_region_width // 2 + middle_region_width, middle_region_step)

        crop_lines = zip([slice(i - horizontal_line_thickness // 2, i - horizontal_line_thickness // 2 + horizontal_line_thickness) for i in horizontal_lines], [middle_region] * slide.shape[0])

    # We save the data for each column of the middle_region
    position = np.zeros(middle_region_width // middle_region_step)
    count = np.zeros(middle_region_width // middle_region_step)
    found = np.zeros(middle_region_width // middle_region_step)

    # For each horizontal cropped line (with thickness horizontal_line_thickness)
    # Notice: Here instead of iterating by columns we iterate by rows
    for i, s in enumerate(crop_lines):  # move from A -> P line by line

        # Notice: s includes (first axis, second axis data)
        # After this step we will have a horizontal line of width (middle_region_width // middle_region_step) and thickness 1
        if horizontal_line_thickness > 1:
            if axis_ap == 0:
                current_values = np.mean(slide[s], axis=1)
            else:
                current_values = np.mean(slide[s], axis=0)
        else:
            current_values = slide[s].squeeze() #for axis_ap==1 there is an single-dimension entryº


        # For each column of the line (i)
        for j in range(len(current_values)):
            if not found[j]: # if sternum has not been found for this column
                if current_values[j] > voxel_treshold: #if value is bigger than a threshold
                    if count[j] == 0:
                        position[j] = i  # We save the first position of the zone as a candidate for the sternum
                    elif count[j] > min_voxels:
                        found[j] = position[j] # if zone bigger the min_voxels, we found sternum position for that column
                    count[j] += 1

                else:
                    count[j] = 0

        if found.all(): #stop if the sternum has been found for all columns
            break

    if not found.any(): # if no sternum has been found
        raise RuntimeError("Sternum not found!")

    #Remove columns where the sternum has not been found (if any)
    #todo: perhaps better throw error here, to ensure a proper result always??
    found = found[found > 0]

    # Compute the mean position of the sternum for all columns
    position = int(round(np.mean(found))) # Median ??

    # Find global A-P axis and direction
    sternum_axis, sternum_dir = DetermineAxisInAnatomicalOrientaion(anatomical_orientation, "A")

    #print(position, axis, dir, found)
    return position, sternum_axis, sternum_dir



