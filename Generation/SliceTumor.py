from . DataProcessing import *
from . ROI import *
import math
import cv2

# Slice a tumor with pre and post information in an image file
def SliceTumor(name, pre, post, mask, input_format, class_type, output_folder, percentage_slices=1.0, slice_axis=0 ):

    fnames = []
    labels = []
    names = []
    slices = []
    info = []

    # Get Bounding Box
    bb = MRILabel2Box(LoadMRISITK(mask), size=(None, None, None))

    # Load input following the input format
    data = ProcessInputCode(pre, post, input_format, data_loader=TryLoadMRI)

    # Check that the size of the data_roi is consistance with the input code
    if len(data) != len(input_format):
        raise RuntimeError("Unexpected number channels for the given input code")

    # Check if data is repeated:
    for d1 in range(len(data) - 1):
        for d2 in range(d1 + 1, len(data)):
            if np.array_equal(data[d1], data[d2]):
                info.append((name,"Warning: Unexpected repeated data ({} == {})! Input format: {} - Num posts: {}".format(d1,d2,input_format,len(post))))
                continue

    # Crop the ROI from the data
    data_roi = GetRoi(image=data, roi=bb, data_loader=TryLoadMRI, postprocessing=None, axis_channel=1)

    # Check that roi shape is the same for all channels
    if any(data_roi[0].shape != ch.shape for ch in data_roi[1:]):
        raise RuntimeError("Unexpected error. Different roi chanels has different shapes!")

    # Get roi shape
    roi_shape = data_roi[0].shape

    # Define the slice for each channel (Cut each channel roi), slide number in slice_ch[slice_axis]
    slice_ch = [slice(0, axis_size) for axis_size in roi_shape]

    # For each slice of the slice_axis
    # for i in range(roi_shape[slice_axis]):
    num_slices = max(math.ceil(roi_shape[slice_axis] * percentage_slices), 1)
    slice_start = roi_shape[slice_axis] // 2 - num_slices // 2
    slice_stop = slice_start + num_slices
    # print(roi_shape[slice_axis], num_slices, slice_start, slice_stop)
    for i in range(slice_start, min(slice_stop, roi_shape[slice_axis])):

        image_name = name + "-" + str(i) + "_" + class_type + ".png"
        image_file = os.path.join(output_folder, image_name)
        if os.path.exists(image_file):
            info.append((name, "Warning: Image file for slice {} already exists!".format(i)))
            continue

        slice_ch[slice_axis] = i  # Definide current slice
        image = [ch[tuple(slice_ch)] for ch in data_roi]
        image = np.stack(image, axis=2)  # opencv color coponen axis is 0

        # Normalize within 8 bits
        min_p = np.min(image)
        if min_p < 0:
            raise RuntimeError("Unexpected min value below zero!")  # this is not expected
        max_p = np.max(image)
        image = ((image.copy() / max_p) * 255).astype(np.uint8)

        # Write image
        cv2.imwrite(image_file, image)

        fnames.append(image_name)
        labels.append(class_type)
        names.append(name)
        slices.append(i)

    return fnames, labels, names, slices, info