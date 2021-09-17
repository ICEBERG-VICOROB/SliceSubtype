import numpy as np
from scipy import ndimage as nd
from scipy.spatial.transform import Rotation
import time

#Use clean_input_mem for an efficient memory process
def GetRois( images,  roi, data_loader=None, preprocessing=None, postprocessing=None, axis_channel=-1, show_progress=False, clean_input_mem=True):

        if show_progress:
            print('', "Processing dataset", end='', flush=True)
            t0 = time.time()

        #check data loader and preprocessing
        if data_loader is not None:
            if not isinstance(data_loader, tuple):
                data_loader = (data_loader,) #comma important! to create a tuple
            if not all(isinstance(dl, list) for dl in data_loader):
                data_loader = tuple([dl if isinstance(dl, list) else [dl] for dl in data_loader])

        if preprocessing is not None:
            if not isinstance(preprocessing, tuple):
                preprocessing = (preprocessing,) #comma important! to create a tuple
            if not all(isinstance(pr, list) for pr in preprocessing):
                preprocessing = tuple([pr if isinstance(pr, list) else [pr] for pr in preprocessing])

        if postprocessing is not None:
            if not isinstance( postprocessing, tuple):
                postprocessing = (postprocessing,)
            if not all(isinstance(po, list) for po in postprocessing):
                postprocessing = tuple([po if isinstance(po, list) else [po] for po in postprocessing])

        #check input data format
        if any([len(images[i]) != len(images[i+1]) for i in range(len(images)-1)]):
            raise RuntimeError("Warning: data length for different images not consistent!")

        if len(images) != len(roi):
            raise RuntimeError("Error: Unconcistent number of images or/and rois!")

        if clean_input_mem:
            images.reverse()

        results = []
        num_images = len(images)
        for i in range(num_images):

            if clean_input_mem:
                img = images.pop()
            else:
                img = images[i]

            #print('.', end='', flush=True)

            #determine number of channels
            if isinstance(img, tuple):
                num_channels = len(img)
                if axis_channel < 0:
                    raise RuntimeError("The axis of the channel component must be specified!")
            else: # (0 means that channel dimension do not exists, if no tuple is used)
                if axis_channel >= 0:
                    num_channels=-1 #unkown
                else:
                    num_channels = 0
                img = (img)

            #load data if needed
            if data_loader is not None:
                if len(data_loader) == 1:
                    img_array = [data_loader[0][0](img_ch, *data_loader[0][1:]) for img_ch in img]
                elif len(data_loader) == len(img):
                    img_array = [data_loader[idx][0](img_ch, *data_loader[idx][1:]) for idx, img_ch in enumerate(img)]
                else:
                    raise RuntimeError("Warning: Unconcistent data_loader data!")
            elif not all(isinstance(img_ch, np.ndarray) for img_ch in img):
                raise RuntimeError("Unkown input data type")

            #if numchannel is unkown, split data in channels
            if num_channels < 0:
                if axis_channel >= len(img_array.shape):
                    raise RuntimeError("Axis channel out of range!")
                # split channels and save them in a tuple
                num_channels = img_array.shape[axis_channel]
                img_array = tuple([ img_array[
                            tuple([ch if i == axis_channel else slice(0,axis_size) for i, axis_size in enumerate(img_array.shape)])
                            ] for ch in range(num_channels)])


            #preprocess data if needed
            if preprocessing is not None:
                if len(preprocessing) == 1:
                    img_array = [preprocessing[0][0](img_ch, *preprocessing[0][1:]) for img_ch in img_array]
                elif len(preprocessing) == len(img_array):
                    img_array = [preprocessing[idx][0](img_ch, *preprocessing[idx][1:]) for idx, img_ch in enumerate(img_array)]
                else:
                    raise RuntimeError("Warning: Unconcistent preprocessing data!")


            #Join channels
            if num_channels >= 1: #if one or more channels is used, stack it (for 1, create 1 new dim)
                #check consistency of sizes
                if any([img_ch.shape != img_array[0].shape for img_ch in img_array]):
                    raise RuntimeError("Error! Image channels have different shapes!")
                img_array = np.stack(img_array, axis=axis_channel) #add option to choose staking axis for channel? by default is 1
            else:
                img_array = img_array[0]

            # add image to the collection
            cropped_img_array = CropRoi(roi[i], img_array, channels_axis=axis_channel if num_channels >= 1 else -1)

            #preprocess data if needed
            if postprocessing is not None:
                if len(postprocessing) == 1:
                        #process all channels at a time (slower)
                        #cropped_img_array = postprocessing[0][0](cropped_img_array, *postprocessing[0][1:])
                        #process single channel
                        post_img = []
                        for ch in range(num_channels): #Post process each channel independely
                            slice_ch = tuple([ch if i == axis_channel else slice(0,axis_size) for i, axis_size in enumerate(img_array.shape)])
                            post_img.append(postprocessing[0][0](cropped_img_array[slice_ch], *postprocessing[0][1:]))
                        cropped_img_array = np.stack(post_img, axis=axis_channel)
                elif len(postprocessing) == len(img_array):
                    if num_channels >= 1:
                        post_img = []
                        for ch in range(num_channels):  #Post process each channel independely
                            slice_ch = tuple([ch if i == axis_channel else slice(0, axis_size) for i, axis_size in enumerate(img_array.shape)])
                            post_img.append(postprocessing[ch][0](cropped_img_array[slice_ch], *postprocessing[ch][1:]))
                        cropped_img_array = np.stack(post_img, axis=axis_channel)
                    else:
                        cropped_img_array = postprocessing[0][0](cropped_img_array, *postprocessing[0][1:])
                else:
                    raise RuntimeError("Warning: Unconcistent postprocessing data!")

            channels = []
            for ch in range(cropped_img_array.shape[axis_channel]):  # Post process each channel independely
                slice_ch = tuple([ch if i == axis_channel else slice(0, axis_size) for i, axis_size in enumerate(cropped_img_array.shape)])
                channels.append(cropped_img_array[slice_ch])

            results.append(tuple(channels))

        return results



def GetRoi(image, roi, data_loader=None, preprocessing=None, postprocessing=None, axis_channel=-1,
            show_progress=False):
    if show_progress:
        print('', "Processing dataset", end='', flush=True)
        t0 = time.time()

    # check data loader and preprocessing
    if data_loader is not None:
        if not isinstance(data_loader, tuple):
            data_loader = (data_loader,)  # comma important! to create a tuple
        if not all(isinstance(dl, list) for dl in data_loader):
            data_loader = tuple([dl if isinstance(dl, list) else [dl] for dl in data_loader])

    if preprocessing is not None:
        if not isinstance(preprocessing, tuple):
            preprocessing = (preprocessing,)  # comma important! to create a tuple
        if not all(isinstance(pr, list) for pr in preprocessing):
            preprocessing = tuple([pr if isinstance(pr, list) else [pr] for pr in preprocessing])

    if postprocessing is not None:
        if not isinstance(postprocessing, tuple):
            postprocessing = (postprocessing,)
        if not all(isinstance(po, list) for po in postprocessing):
            postprocessing = tuple([po if isinstance(po, list) else [po] for po in postprocessing])

    # determine number of channels
    if isinstance(image, tuple):
        num_channels = len(image)
        if axis_channel < 0:
            raise RuntimeError("The axis of the channel component must be specified!")
    else:  # (0 means that channel dimension do not exists, if no tuple is used)
        if axis_channel >= 0:
            num_channels = -1  # unkown
        else:
            num_channels = 0
        img = (image)

    # load data if needed
    if data_loader is not None:
        if len(data_loader) == 1:
            img_array = [data_loader[0][0](img_ch, *data_loader[0][1:]) for img_ch in image]
        elif len(data_loader) == len(image):
            img_array = [data_loader[idx][0](img_ch, *data_loader[idx][1:]) for idx, img_ch in enumerate(image)]
        else:
            raise RuntimeError("Warning: Unconcistent data_loader data!")
    elif not all(isinstance(img_ch, np.ndarray) for img_ch in image):
        raise RuntimeError("Unkown input data type")

    # if numchannel is unkown, split data in channels
    if num_channels < 0:
        if axis_channel >= len(img_array.shape):
            raise RuntimeError("Axis channel out of range!")
        # split channels and save them in a tuple
        num_channels = img_array.shape[axis_channel]
        img_array = tuple([img_array[
                               tuple([ch if i == axis_channel else slice(0, axis_size) for i, axis_size in
                                      enumerate(img_array.shape)])
                           ] for ch in range(num_channels)])

    # preprocess data if needed
    if preprocessing is not None:
        if len(preprocessing) == 1:
            img_array = [preprocessing[0][0](img_ch, *preprocessing[0][1:]) for img_ch in img_array]
        elif len(preprocessing) == len(img_array):
            img_array = [preprocessing[idx][0](img_ch, *preprocessing[idx][1:]) for idx, img_ch in
                         enumerate(img_array)]
        else:
            raise RuntimeError("Warning: Unconcistent preprocessing data!")

    # Join channels
    if num_channels >= 1:  # if one or more channels is used, stack it (for 1, create 1 new dim)
        # check consistency of sizes
        if any([img_ch.shape != img_array[0].shape for img_ch in img_array]):
            raise RuntimeError("Error! Image channels have different shapes!")
        img_array = np.stack(img_array,
                             axis=axis_channel)  # add option to choose staking axis for channel? by default is 1
    else:
        img_array = img_array[0]

    # add image to the collection
    cropped_img_array = CropRoi(roi, img_array, channels_axis=axis_channel if num_channels >= 1 else -1)

    # preprocess data if needed
    if postprocessing is not None:
        if len(postprocessing) == 1:
            # process all channels at a time (slower)
            # cropped_img_array = postprocessing[0][0](cropped_img_array, *postprocessing[0][1:])
            # process single channel
            post_img = []
            for ch in range(num_channels):  # Post process each channel independely
                slice_ch = tuple([ch if i == axis_channel else slice(0, axis_size) for i, axis_size in
                                  enumerate(img_array.shape)])
                post_img.append(postprocessing[0][0](cropped_img_array[slice_ch], *postprocessing[0][1:]))
            cropped_img_array = np.stack(post_img, axis=axis_channel)
        elif len(postprocessing) == len(img_array):
            if num_channels >= 1:
                post_img = []
                for ch in range(num_channels):  # Post process each channel independely
                    slice_ch = tuple([ch if i == axis_channel else slice(0, axis_size) for i, axis_size in
                                      enumerate(img_array.shape)])
                    post_img.append(postprocessing[ch][0](cropped_img_array[slice_ch], *postprocessing[ch][1:]))
                cropped_img_array = np.stack(post_img, axis=axis_channel)
            else:
                cropped_img_array = postprocessing[0][0](cropped_img_array, *postprocessing[0][1:])
        else:
            raise RuntimeError("Warning: Unconcistent postprocessing data!")

    channels = []
    for ch in range(cropped_img_array.shape[axis_channel]):  # Post process each channel independely
        slice_ch = tuple([ch if i == axis_channel else slice(0, axis_size) for i, axis_size in
                          enumerate(cropped_img_array.shape)])
        channels.append(cropped_img_array[slice_ch])

    return tuple(channels)


# roi: [(x1,x2), (y1,y2), ..., (n1,n2))
# affine_transform: (n+1)x(n+1) affine matrix transformation
# trans_margin_px: addition pixel added in the big_roi (croped region where the transformation take place)
#                  this will help to better interpolate the affine transformed result (use whole image: -1)
# spline_inter_order: spline interpolation order for the affine transform [0,5]
# channel_axis: axis number of the channels/compnents, -1 if not used
# trans_single_ch: apply transfor to each channel independely each channel independly (IMPORTANT: Seem both provide same result but True much faster)

def CropRoi(roi, array, affine_transform=None, spline_inter_order=3, trans_margin_px=0, channels_axis=-1,
            trans_single_ch=True):  # roi: [(x1,x2), (y1,y2), etc]

    # In this code the roi is cropped from the array. If a transformation is applied, the roi is transformed and
    # the transformed roi cropped zone is returned in the shape of the original roi
    #
    #                   ...............
    #   ___             ......___......       ___
    #  |  |             ..../   /......      |..|
    #  |  |  +  T       .../   /.......  =>  |..|
    #  ---              ..----.........      ---
    #  ROI   AFFINE     ...............
    #        TRANS

    # For efficiency first the image is cropped in a big roi, which include de original roi and transformed roi
    # therefore the affine transformation is not applied for the whole image, only for this big cropped zone
    # If trans_margin_px is -1, the whole image is used
    #
    #                   ................
    #   ___      ___    .  ___ ___ .....
    #  |  |    /   /    . |  /|  / .....
    #  |  | + /   /     . |/  |/   .....
    #  ---   ----       . ----     .....
    #  ROI    T-ROI     ................
    #

    # Notice: Currently scale transformed rois will be return interpolated to the original roi size
    #         This can be changed by returning in the shape of the transformed roi (checkingn the transformed roi x,y and z lengths)


    # TODO: the code can be further optimaze moving the roi inside the transformed_roi region in case is outside
    #       Notice: due to translation the roi can be outisde of the axis aligned roi defined by the transformed roi
    #       so, currenly the big roi include both the original roi and transformed roi. This changes should also consider
    #       reside the roi in case of scaling!

    # Multichannel solution
    # Currenly we apply the affine transformation for the multichannel array (option A)
    # todo: check it is better to apply the transformation to each of the channels individualy (option B)

    # save array shape (without the channel axis)
    array_shape = array.shape[:channels_axis] + array.shape[(channels_axis + 1):] if channels_axis >= 0 else array.shape  # it can be summ beacouse is tuple

    if trans_margin_px < 0:
        trans_margin_px = np.max(array_shape)

    if spline_inter_order < 0 or spline_inter_order > 5:
        raise RuntimeError("spline interpolation order parameter out of range [0-5]")

    # need to conder axis channel >= 0
    # if len(roi) != len(array.shape)  or (affine_transform is not None and len(roi)+1 != len(affine_transform)):
    #    raise RuntimeError("Input parameters have inconsistent lengths")

    if affine_transform is not None:

        # compute transformed roi
        p1, p2 = tuple(np.array(roi).transpose())
        roi_mat = np.array(np.meshgrid(*roi)).T.reshape(-1,
                                                        3)  # get 2^n points definition EX: xyz->8points (cube)
        roi_mat = np.pad(roi_mat, [(0, 0), (0, 1)],
                         constant_values=1)  # add 0 pad at the end [1,2,3] -> [1,2,3,0]
        transformed_roi_mat = (
                    affine_transform @ roi_mat.transpose()).transpose()  # Transform points (affine_transform @ A.transpose()).transpose()

        # big roi: axis oriented roi where the transformed roi fits inside
        big_roi_p1, big_roi_p2 = np.floor(transformed_roi_mat.min(axis=0)[:-1]).astype('int'), np.ceil(
            transformed_roi_mat.max(axis=0)[:-1]).astype('int')  # minimum point, and maximum point

        # check that original roi is included: (because of translation original roi could be outside transformed roi)
        big_roi_p1, big_roi_p2 = np.minimum(big_roi_p1, p1), np.maximum(big_roi_p2, p2)
        # increse 1 + [trans_margin_px] to account for decimal errors and [interpolation quality] and check for maximum image size bounders
        big_roi_p1, big_roi_p2 = np.maximum(big_roi_p1 - 1 - trans_margin_px, 0), np.minimum(
            big_roi_p2 + 1 + trans_margin_px, array_shape)

        br = list(zip(big_roi_p1,
                      big_roi_p2))  # covert to orginal roi format [(x1,x2),(y1,y2),ect)] [:-1] all minus last one

        br_array = CropArray(array, br, channels_axis=channels_axis)
        # Add translation before the centering and after it recentering
        br_affine_transform = TranslationMat(-big_roi_p1) @ affine_transform @ TranslationMat(
            big_roi_p1)  # get transfrom for the big roi
        br_roi = [(v1 - disp, v2 - disp) for (v1, v2), disp in zip(roi, big_roi_p1)]

        if channels_axis >= 0:

            if not trans_single_ch:

                # comment this for test
                raise RuntimeError("NOTICE! This option is slower an test do not show better results than option B")

                # A. Apply a 1 more dimension transformation (for channel dimension) EX: 3D affine (4x4) -> 4D affine (5x5)
                # account for the channel axis, which is not transformed (add a row, column with 1 on the axis value)
                # EX: channels_axis = 1                        #
                # 0.2  0.8  3         0.2  0.8  3         0.2  0  0.8  3
                # 0.8  0.2  1   =>  # 0    0    0 #  =>   0    1  0    0
                # 0    0    1         0.8  0.2  1         0.8  0  0.2  1
                #                     0    0    1         0    0  0    1
                #                                              #
                zero_array = np.zeros(len(br_affine_transform))  # for row
                unit_array = np.array(
                    [1 if i == channels_axis else 0 for i in range(len(br_affine_transform) + 1)])  # for column
                br_affine_transform = np.stack(
                    [*br_affine_transform[:channels_axis], zero_array,
                     *br_affine_transform[channels_axis:]])  # add row
                br_affine_transform = np.stack([*br_affine_transform[:, :channels_axis].transpose(), unit_array,
                                                *br_affine_transform[:, channels_axis:].transpose()],
                                               axis=1)  # add column

                # https://het.as.utexas.edu/HET/Software/Scipy/generated/scipy.ndimage.interpolation.affine_transform.html
                br_array_transformed = nd.affine_transform(br_array, br_affine_transform, mode='constant',
                                                           order=spline_inter_order)

            else:
                # IMPORTANT: after several test seems that bot provide same result but this one is faster
                # B. Transform each channel separetly (more than 4 times faster but not sure that provides the same consistent result))
                br_array_transformed = np.zeros_like(br_array)
                # br_array_transformed = [] #append and stack method
                for ch in range(array.shape[channels_axis]):
                    ch_slice = tuple([ch if i == channels_axis else slice(0, ch_size) for i, ch_size in
                                      enumerate(array.shape)])
                    br_array_transformed[ch_slice] = nd.affine_transform(br_array[ch_slice],
                                                                         br_affine_transform, mode='constant',
                                                                         order=spline_inter_order)
                    # br_array_transformed.append(nd.affine_transform(br_array[ch_slice], br_affine_transform, mode='constant', order=spline_inter_order))
                # br_array_transformed = np.stack(br_array_transformed, axis=channels_axis)

        else:

            br_array_transformed = nd.affine_transform(br_array, br_affine_transform, mode='constant',
                                                       order=spline_inter_order)

        crop_array_transformed = CropArray(br_array_transformed, br_roi, channels_axis=channels_axis)

    else:  # if no transformation is needed
        crop_array_transformed = CropArray(array, roi, channels_axis=channels_axis)

    return crop_array_transformed



def CropArray(image_array, roi, channels_axis=-1):

    # Create crop slice
    crop_slice = tuple([slice(start, end) for start, end in roi])

    if channels_axis >= 0:
        crop_slice = crop_slice[0:channels_axis] + (slice(0, image_array.shape[channels_axis]),) + crop_slice[channels_axis:]

    #crop image
    cropped_array = image_array[crop_slice]

    return cropped_array

def TranslationMat(trans):
    mat = np.eye(len(trans)+1)
    mat[:len(trans),-1] = trans #last line
    return mat

def ReflectionMat(reflect):
    mat = np.eye(len(reflect)+1)
    for i in np.where(reflect):
        mat[i,i] = -1
    return mat

def RotationMat(rotation_obj):
    if not isinstance(rotation_obj, Rotation):
        raise RuntimeError("Rotation object must be of type Rotation!")
    rot_mat = rotation_obj.as_matrix()
    mat = np.eye(len(rot_mat)+1)
    mat[:len(rot_mat),:len(rot_mat)] = rot_mat
    return mat


