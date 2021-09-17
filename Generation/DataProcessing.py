from scipy import ndimage as nd
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import random
import os

#a = np.array([[[0,0,0,0], [0,1,1,0], [0,1,1,1]], [[0,0,0,0], [0,1,0,0], [0,0,0,0]], [[0,0,0,0],[0,1,0,0],[0,0,0,0]]])
#c = np.where(a != 0)
#[(min(c[i]), max(c[i])) for i in range(0,3)]
# Others: https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array

#format [(start_dim1, end_dim1), (start_dim2, end_dim2), ... ]
def Label2BoundingBox(label):
    nonzero_pos = np.where(label != 0)
    bb = [(min(nonzero_pos[i]), max(nonzero_pos[i])+1) for i in range(0, len(label.shape))]
    return bb


def MRILabel2BoundingBox(label):
    label_array =  sitk.GetArrayFromImage(label)
    nonzero_pos = np.where(label_array != 0)
    bb = [(min(nonzero_pos[i]), max(nonzero_pos[i])+1) for i in range(0, len(label_array.shape))]
    return bb



def MRILabel2Box(label, size = (None, None, None), name="Unkown"):
    label_image_size = label.GetSize()[::-1]
    roi = MRILabel2BoundingBox(label)

    if len(size) != len(label_image_size):
        print("{}: Error Box creation size match".format(name))
        return None

    for i in range(0, len(size)):
        if size[i] is not None:

            if label_image_size[i] < size[i]:
                print("{}: Error MRI image to small".format(name))
                return None

            if  size[i] is not None and roi[i][1] - roi[i][0] > size[i]:
                print("{}: Discarted too big roi!".format(name))
                return None

            delta = size[i] - (roi[i][1] - roi[i][0])
            delta = (delta // 2, delta - delta // 2)

            delta_dif1 = max(delta[0] - roi[i][0], 0) #out of range in x0 border (0 if inside)   [<-|--o--->]
            delta_dif2 = max(roi[i][1] + delta[1] - label_image_size[i], 0 ) #out of range in xmax border (0 if inside)

            #redifine delta_x (Condition: roi smaller than image!!)
            roi[i] = (roi[i][0] - (delta[0] + delta_dif2 - delta_dif1), roi[i][1] + (delta[1] + delta_dif1 - delta_dif2))

    return roi



def BoundingBox2Label(bb, array_size):
    label = np.zeros(array_size)
    bb_slice = [slice(start, end) for start, end in bb]
    label[bb_slice] = 1
    return label



def CropMRI(image, roi):

    #Transform image to array
    image_array = sitk.GetArrayFromImage(image)

    #crop image array
    cropped_array = CropArray(image_array, roi)

    #backup image info
    spacing, origin, direction = image.GetSpacing(), image.GetOrigin(), image.GetDirection()

    #Create new sitk image
    image = sitk.GetImageFromArray(cropped_array)

    # Restore info
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)

    return image


def CropArray(image_array, roi, channels_axis=-1):

    # Create crop slice
    crop_slice = tuple([slice(start, end) for start, end in roi])

    if channels_axis >= 0:
        crop_slice = crop_slice[0:channels_axis] + (slice(0, image_array.shape[channels_axis]),) + crop_slice[channels_axis:] #to add tuple (,) comma at the end important!

    #crop image
    cropped_array = image_array[crop_slice]

    return cropped_array




def ProcessInputCodeList(pre_list, post_list, input, data_loader=None):
    return [ ProcessInputCode(pre_item, post_item, input, data_loader=data_loader) for pre_item, post_item in zip(pre_list, post_list) ]


def ProcessInputCode(pre, post, input, data_loader=None):
    input_result = []
    pre = TryLoadData(pre, data_loader=data_loader)
    post = [TryLoadData(p, data_loader=data_loader) for p in post]
    if not isinstance(pre, np.ndarray) or not all(isinstance(p, np.ndarray) for p in post) or pre.shape != post[
        0].shape or any(post[0].shape != post[i].shape for i in range(1, len(post))):
        raise RuntimeError("Data format error while processing input")

    for code in input:
        # Parser channel code
        channel = eval(code)
        input_result.append(channel)

    return tuple(input_result)









#LoadMRI
def mask_image(im):
    return im > 0





# Try to load MRI as a file, sitk or np.ndarray to np.ndarray
def TryLoadMRI(element):
    if isinstance(element, np.ndarray):
        return element
    elif isinstance(element, sitk.Image):
        return Sitk2NP(element)
    elif os.path.isfile(element):
        return LoadMRI(element)
    else:
        raise RuntimeError("Unkown data format")

def LoadMRI(file):
    #print(file)
    #return nib.load(file).get_data().astype('float32')

    image = sitk.ReadImage(file)
    return np.array(sitk.GetArrayFromImage(image))


def Sitk2NP(mri):
    return np.array(sitk.GetArrayFromImage(mri), dtype="float32")

def LoadMRISITK(file):
    #print(file)
    #return nib.load(file).get_data().astype('float32')

    return sitk.ReadImage(file)


def TryLoadData(items, data_loader = None):
    # check data loader and preprocessing
    if data_loader is not None:
        if not isinstance(data_loader, list):
            data_loader = [data_loader]  # comma important! to create a tuple

        if isinstance(items, list):
            return [data_loader[0](item, *data_loader[1:]) for item in items]
        else:
            return data_loader[0](items, *data_loader[1:])
    else:
        return items




def NormalizeMAIASITK(image, norm_type='standard', use_mask=False, datatype=np.float32):

    image_array = sitk.GetArrayFromImage(image)

    # crop image array
    normalized_array = NormalizeMAIA(image_array, norm_type=norm_type, use_mask=use_mask, datatype=datatype)

    # backup image info
    spacing, origin, direction = image.GetSpacing(), image.GetOrigin(), image.GetDirection()

    # Create new sitk image
    image = sitk.GetImageFromArray(normalized_array)

    # Restore info
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)

    return image



def NormalizeMAIA(image, norm_type='standard', use_mask=False, datatype=np.float32):

    if not isinstance(image, tuple):
        image = [image]

    out = []
    for i in range(len(image)):
        mask = np.copy(image[i] > 0)

        out.append(image[i].astype(dtype=datatype))
        # image = np.pad(image, ((0,0), (16, 16), (16, 16), (16, 16)))

        if norm_type == 'standard':
            #list_nonzero = np.nonzero(out[i])
            out[i] = out[i] - out[i][mask].mean()
            out[i] = out[i] / out[i].std()
            #out[i] = out[i] / out[i][mask].std()

        if norm_type == 'zero_one':
            min_int = abs(image[i].min())
            max_int = image[i].max()
            if image[i].min() < 0:
                image[i] = image[i] + min_int
                image[i] = image[i] / (max_int + min_int)
            else:
                image[i] = (image[i] - min_int) / max_int

            if use_mask:
                out[~mask] = 0
        # do not apply normalization to non-brain parts
        # im[mask==0] = 0
    if len(out) > 1:
        out = tuple(out)
    else:
        out = out[0]

    return out


# Normalization with outlier removal
# Normalize the image removing the values smaller and bigger than the min and max indicated percentage of all values
def Normalization( image, min_outlier_percentage=0.001, max_outlier_percentage=0.001):

    #Convert to float
    image = image.astype(np.float32)

    # Sort by intensity values
    value_sort = np.sort(image.flatten())

    #Find the minimum and maximum values (discarting the specified percentage)
    min_value = value_sort[int( min_outlier_percentage * len(value_sort))]
    max_value = value_sort[-int( max_outlier_percentage * len(value_sort))]

    #normalize
    image = 1.0 * (image - min_value) / (max_value - min_value)

    #In case of values out of normalzed range (possible??), rectify
    image[image > 1.0] = 1.0
    image[image < 0.0] = 0.0

    return image


def ZScoreNormalization(image):
    image = (image - np.mean(image)) / np.std(image)
    return image


def ResizeImage(image, spacing, target_spacing, order = 1):

    #Compute scale factor based on the spacing difference
    scale_spacing = spacing / target_spacing

    # Interpolate image (n order interpolation)
    image = nd.interpolation.zoom(image, scale_spacing, order=order)

    return image


def ResizeImage2(image, target_size, order = 1):

    if len(image.shape) != len(target_size):
        raise RuntimeError("Wrong input target_size length!")

    if any(v is None for v in target_size):
        target_size = tuple([ image.shape[i] if target_size[i] is None else target_size[i] for i in range(len(target_size))])

    #Compute scale factor based on the spacing difference
    scale = np.array(target_size) / np.array(image.shape)

    # Interpolate image (n order interpolation)
    image = nd.interpolation.zoom(image, scale, order=order)

    #crop or pad in case
    difference = np.array(target_size) - np.array(image.shape)

    if any(difference > 0):
        image = image[tuple([slice(0, s) for s in target_size])]

    if any(difference < 0):
        np.pad(image, tuple([v if v < 0 else 0 for v in difference]), 'constant')

    return image


def NormalizationAdvance(image):

    return ZScoreNormalization(Normalization(image))



class MRIData(np.ndarray):

    def __new__(cls, input_array, spacing=None, origin=None, direction=None, header=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.spacing = spacing
        obj.origin = origin
        obj.direction = direction
        obj.header = header
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.spacing = getattr(obj, 'spacing', None)
        self.origin = getattr(obj, 'origin', None)
        self.direction = getattr(obj, 'direction', None)
        self.header = getattr(obj, 'header', None)