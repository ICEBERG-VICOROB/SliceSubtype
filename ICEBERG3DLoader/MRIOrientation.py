# MRI Orientation tools
# Author: Joel Vidal <jolvid@gmail.com>
# Date: 20/05/20
# Version: 0.3

import numpy as np
import SimpleITK as sitk
import math

# Todo: Need additional comments for some functions
#       Add/Improve comments


"""
 Understanding the anatomical space and coordinates
 ==================================================
 When the MRI is taken the position of the patient w.r.t the machine determines the orientation of the data
 on the x,y,z coordinates. On the other side, we define the anatomical space, which is a interpretation of physical
 space related to medical practice. This space is defined w.r.t the patient by the anatomical directions defined by
 the terms: Left/Right(like arms), Anteriori/Posteriori (in front of/behind) and Inferiori/Superiori (bottom/top)
 Then, based on the position of the patient we associate the anatomical directions with the physical coordinates x,y,z.
 For example, associate the x-axis with the left->right (L->R, for short), y-axis with I->S and z-axis with P->A
 This will be written, for short, as LIP, first letter for x, second for y and third for z. Different anatomical
 positions with respect to the data coordinates will result in different associations (ex: LPI, RAI, etc)
 
 This anatomic space information must be defined manually as the MRI machine/software do not know the respective
 position of the patient respect to the data coordinate space (referred as physical space) and the user must
 specify the right one and save it in the metadata alongside wth the image. So the further viewers can associate
 what are the anatomical directions of each data axis.

 It is possible to represents the data with respect to an absolut coordinate system (i.e. world) using a fixed
 convention where the x, y and z positive axis directions are associated with a fixed anatomical directions.
 In this case, any anatomic association can be represented by a mathematical transformation (rigid rotation SO(3))
 w.r.t to the fixed convention. Therefore, is common to define the anatomical directions using a mathematical
 definition like a 3x3 rotation matrix or a quaternion representation. We must notice again that this mathematical
 transformations are defined with respect to a fixed convention which can vary for different data interpreters.

 Some examples of anatomical conventions:
 Used by DICOM or ITK:         +x = R -> L     +y = A -> P    +z = I -> S     (the other order is negative axis)
 Used by NIFTI or ITK-SNAP:    +x = L -> R     +y = P -> A    +z = I -> S     (the other order is negative axis)

 For example: In a NIFTI file the anatomical space (also named Orientation/Direction) is defined by a quaternion
 transformation (with metadata tags quatern_b,  quatern_c,  quatern_d) with respect to the NIFTI anatomical
 convention (Then defined by the matrix identity) LPI.
 Same anatomical definition of the space will be a different mathematical transformation for a different interpreter
 like ITK (due to the use of a different convention).

 For example to specify that an specific data x, y, z-axis are associated to the ASL anatomical directions:

         DICOM: [[-0.  0. -1.]         NIFTI: [[ 0.  0.  1.]
                 [ 1. -0.  0.]                  [-1.  0.  0.]
                 [ 0. -1.  0.]]                 [ 0. -1.  0.]]

 Anatomical Conventions:
 Other anatomical spaces are defined as mathematics transformations w.r.t the convention

          Coor. conventions                Anatomical Space
    ITKSNAP/NIFTI      DICOM/ITK*
                                                             Toes
                                                             [I]
     +y                                    [A]          _    :  _
      ^            +x <-------|             |         /  \  : /  \
      |                     / |             |        {    \:_|    \ <- Breast representation
      |                   /   |             |        |    :       |
      !-------> +x     |/_    v +y         [P]    [L]----:---------[R]
     /                +z                                [S]
  |/_                                                 Head
 +z

 *With sitk, the DICOM/ITK convention is used

 NOTES:
 GetArrayViewFromImage?? CHECK : https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/21_Transforms_and_Resampling.html
 Notice that each image data corresponds to an specific orientation (depending of how the patient was located with respect to machine's x,y,z axis)
 For example if patient data y axis (np.array) is from R->L anatomically then its orientation must be xRx
 Orientations must follow the right-hand rule? (https://en.wikipedia.org/wiki/Right-hand_rule)
 If this orientation want to be changed (ex: to align x-axis to R->L) the data array must be transformed

 LINKS:
 http://itksnap.org/pmwiki/pmwiki.php?n=Documentation.DirectionMatrices
 https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/quatern.html
"""


# Get the direction matrix
def GetDirectionMatrix(image):
    return np.array(image.GetDirection()).reshape(3,3)

# Get direction
def GetDirection(image):
    return np.array(image.GetDirection())

# Get Voxel2World transformation matrix
def GetVoxel2WorldMatrix(orientation, spacing, origin = [0,0,0]):
#todo: concept, not tested
    res = np.eye(4)
    # [..<-z,y,x]
    res[:3,:3] = np.multiply(GetMatrixFromOrientation(orientation), spacing)
    res[:3, 3] = origin
    return res

#Position values in image goes from 0 to n-1
def GetWorldCoordinates(input, orientation, spacing, origin):

    # Check if input is 3x1 or 4x1
    if len(input) != 4:
        raise RuntimeError("Unexpected Input format. It must be a 4x1 array")

    # Get voxel to world homohenous matrix
    v2w_matrix = GetVoxel2WorldMatrix(orientation, spacing, origin)
    res = np.dot(v2w_matrix, input )
    return res


def GetWorldCoordinatesVec( vector, orientation, spacing):

    #Check if input is 3x1 or 4x1
    if len(vector) == 3:
        vector = np.array(list(vector) + [0])
    else:
        raise RuntimeError("Unexpected input format. It must be a 3x1 vector")

    return GetWorldCoordinates(vector, orientation, spacing, origin=[0,0,0])[:3] #Origin is not use for vector calculation

#Position values in image goes from 0 to n-1
def GetWorldCoordinatesPos( position, orientation, spacing, origin):

    # Check if input is 3x1 or 4x1
    if len(position) == 3:
        position = np.array(list(position) + [1])
    else:
        raise RuntimeError("Unexpected input format. It must be a 3x1 vector")

    return GetWorldCoordinates(position, orientation, spacing, origin=origin)[:3]


# ANATOMICAL CONVENTIONS

#DICOM/ITK               ITKSNAP/NIFTI
# X = R -> L              X = L -> R
# Y = A -> P              Y = P -> A
# Z = I -> S              Z = I -> S

#Used in sitk (orientation matrix is based on this anatomical convention)
anatomic_DICOM = { 0: {1: "R", -1: "L"},
                   1: {1: "A", -1: "P"},
                   2: {1: "I", -1: "S"},
                   "R": np.array([1, 0, 0]),
                   "L": np.array([-1, 0, 0]),
                   "A": np.array([0, 1, 0]),
                   "P": np.array([0, -1, 0]),
                   "I": np.array([0, 0, 1]),
                   "S": np.array([0, 0, -1])}


#Used for other coordinate conventions (ITKSNAP and NIFTI)  (used with the sitk orientation mat will give WRONG anatomical orientation)
anatomic_NIFTI = { 0: { 1: "L", -1: "R"},
                   1: { 1: "P", -1: "A"},
                   2: { 1: "I", -1: "S"},
                   "L": np.array([1, 0, 0]),
                   "R": np.array([-1, 0, 0]),
                   "P": np.array([0, 1, 0]),
                   "A": np.array([0, -1, 0]),
                   "I": np.array([0, 0, 1]),
                   "S": np.array([0, 0, -1])}


#Functions for Reading MRI Orientation/Direction


def CheckMatrixisOrthonormal(mat):
    if not isinstance(mat, np.ndarray) or mat.shape != (3,3):
        raise RuntimeError("Wrong matrix format")
    if not np.array_equal(np.dot(mat.transpose(), mat), np.eye(3)):  # Check the transpose by matrix is identity (orthonormal propiety)
        return False
    return True


def CheckMatrixFollowRigthHandRule(mat):
    if not isinstance(mat, np.ndarray) or mat.shape != (3,3):
        raise RuntimeError("Wrong matrix format")
    # Check it follow the right hand rule
    if not np.array_equal(np.cross(mat[:, 0], mat[:, 1]), mat[:, 2]):  # Check right hand rule by cross product
        return False
    return True



def CheckValidOrientation(orientation):

    # Check it has the rigth format
    if isinstance(orientation, str):
        if (len(orientation) != 3 or not ("R" in orientation or "L" in orientation) or\
                not ("A" in orientation or "P" in orientation) or not ("I" in orientation or "S" in orientation)):
            raise RuntimeError("Wrong anatomical orientation format.")

    if isinstance(orientation, np.ndarray):
        if orientation.shape != (9,) and orientation.shape != (3, 3):
            raise RuntimeError("Wrong length orientation format. It must be a 3x3 matrix or a 9 length row-wise vector")

    elif isinstance(orientation, (list, tuple)):
        if len(orientation) != 9:
            raise RuntimeError("Wrong length orientation format. It must be a 9 length row-wise list/tuple")

    else:
        RuntimeError("Unknown orientation format.")

    #Check if follow right hhand rules
    mat = GetMatrixFromOrientation(orientation)

    if not CheckMatrixisOrthonormal(mat):
        RuntimeError("Orientation matrix must be orthonormal")

    if not CheckMatrixFollowRigthHandRule(mat):
        RuntimeError("Orientation must follow the right hand rule")

    return False


def GetCurrentAnatomicalOrientation(image):
    return GetAnatomicalOrientationFromMatrix(GetDirectionMatrix(image))



def GetAnatomicalOrientationFromMatrix(matrix):

    if not CheckMatrixisOrthonormal(matrix):
        RuntimeError("Orientation matrix must be orthonormal")

    if not CheckMatrixFollowRigthHandRule(matrix):
        RuntimeError("Orientation must follow the right hand rule")

    # round the values to integers in case there is small variations Ex: 2.051034e-10
    axis_x = np.rint(matrix[:, 0])
    axis_y = np.rint(matrix[:, 1])
    axis_z = np.rint(matrix[:, 2])

    x_i = np.where(axis_x)[0][0]
    y_i = np.where(axis_y)[0][0]
    z_i = np.where(axis_z)[0][0]



    orentation = (anatomic_DICOM[x_i][axis_x[x_i]],
                 anatomic_DICOM[y_i][axis_y[y_i]],
                 anatomic_DICOM[z_i][axis_z[z_i]])

    orientation_str = ''.join(orentation)

    return orientation_str


def GetAnatomicalOrientationFromOrientation(orientation):

    # Check orientation format
    CheckValidOrientation(orientation)

    if isinstance(orientation, str):
        anat_orientation_str = orientation
    else:
        #Transform to matrix
        mat = GetMatrixFromOrientation(orientation)
        anat_orientation_str = GetAnatomicalOrientationFromMatrix(mat)

    return anat_orientation_str


def GetMatrixFromAnatomicalOrientation(orientation): #, anatomical_convention=anatomic_DICOM
    #Set dicom anatomical comvention (used by sitk)
    anatomical_convention = anatomic_DICOM

    # check input
    if not isinstance(orientation, str) or len(orientation) != 3 and not ("R" in orientation or "L" in orientation) and\
           not ("A" in orientation or "P" in orientation) and not ("I" in orientation or "S" in orientation):
        raise RuntimeError("Uknown orientation parameter format.")
    #if not isinstance(anatomical_convention, dict) or not all( l in anatomical_convention and
    #       isinstance(anatomical_convention[l], np.ndarray) and anatomical_convention[l].shape == (3,) and
    #       abs(sum(anatomical_convention[l])) == 1 for l in ["R","L","A","P","I","S"]):
    #    raise RuntimeError("Wrong anatomical convention format.")
    mat = np.zeros((3,3))
    for i, axis in enumerate(orientation):
        mat[:,i] = anatomical_convention[axis]

    return mat



def GetDirectionFromOrientation(orientation):
    GetMatrixFromOrientation(orientation)



# Works with row-wise matrix and anatomical terms
def GetMatrixFromOrientation(orientation):

    if isinstance(orientation, str) and len(orientation) == 3 and ("R" in orientation or "L" in orientation) and \
            ("A" in orientation or "P" in orientation) and ("I" in orientation or "S" in orientation):
        mat = GetMatrixFromAnatomicalOrientation(orientation)
    elif isinstance(orientation, np.ndarray) and orientation.shape == (9,) :
        if abs(sum(orientation[0:3])) != 1 or abs(sum(orientation[3:6])) != 1 or abs(sum(orientation[6:9])) != 1:
                raise RuntimeError("Wrong orientation format. Each column-wise vector must be normalized.")
        mat = orientation.reshape(3,3)
    elif isinstance(orientation, (list, tuple)) and len(orientation) == 9:
        if abs(sum(orientation[0:3])) != 1 or abs(sum(orientation[3:6])) != 1 or abs(sum(orientation[6:9])) != 1:
            raise RuntimeError("Wrong orientation format. Each column-wise vector must be normalized.")
        mat = np.array(orientation).reshape(3,3)
    elif isinstance(orientation, np.ndarray) or orientation.shape != (3, 3):
        if abs(sum(orientation[:, 0])) != 1 or abs(sum(orientation[:, 1])) != 1 or abs(sum(orientation[:, 2])) != 1:
            raise RuntimeError("Wrong matrix orientation format. Each column must be normalized.")
        mat = orientation
    else:
        raise RuntimeError("Unknown orientation format")


    return mat



# Transformation Functions

# Transforms MRI Orientation
# This function transform the data and change the direction information accordingly
# NOTICE: If the data have wrong direction information use SetDirection instead
def TransformMRIOrientation(image, orientation, verbose = True):


    # From current -> world  w^T_c
    # Notice the basis transformation of the vectors
    #  | wx1 wx2     wxn |     | v11 v21 v31 |   | cx1 cx2 ... cxn |
    #  | wy1 wy2 ... wyn |  =  | v12 v22 v32 | * | cy1 cy2 ... cyn |
    #  | wz1 wz2     wzn |     | v13 v23 v33 |   | cz1 cz2 ... czn |
    #    coord on w                w^T_c              coord on c
    #                       basis transformation
    #
    # Notice that columns vector v1,v2 and v3 are the new basis of the coordinates
    # and vector rows (v11,v21,v31), (v12,v22,v32) and (v13,v23,v33) defines the transformation from the standard basis
    current_mat = GetDirectionMatrix(image)

    # From target -> world w^T_t
    target_mat = GetMatrixFromOrientation(orientation)

    #Show information
    current_anatomical_space = GetAnatomicalOrientationFromMatrix(current_mat)
    target_anatomical_space = GetAnatomicalOrientationFromMatrix(target_mat)
    if verbose:
        print(" Transforming from <{}> to <{}>".format(current_anatomical_space, target_anatomical_space))

    #Transfomration from current -> target    (w^T_t)^(-1) * w^T_c = t^T_w * w^T_c =  t^T_c
    transformation = np.dot(target_mat.transpose(), current_mat)

    # Find the new axis for each dimension (the rows of the transformation matrix)
    axis_x, axis_y, axis_z = transformation[0, :], transformation[1, :], transformation[2,:]

    if sum(axis_x != 0) == 1 and sum(axis_y != 0) == 1 and sum(axis_z != 0) == 1:

        # Get the new position of each axis where x=0, y=1, z=2
        # Ex: New position of axis x is axis z:  x_i=2
        x_i, y_i, z_i   = np.where(axis_x)[0][0], np.where(axis_y)[0][0], np.where(axis_z)[0][0]

        # Direction of the nex axis.
        # Position is already correct only need to add (as they keep the right position)
        # Ex: [0,0,-1] + [0,1,0] + [1,0,0] = [1,1,-1]
        # Ex: Axis x new position (axis z) direction is reversed so axis_dir[2] = -1
        axis_idir = np.array([axis_x[x_i], axis_y[y_i],  axis_z[z_i]], dtype='int')

        # Transform image to array
        # Usually matrix are in sahpe (x,y,z) where slicing is done [dimn...dim0] so [z,x,y] but sitk provides inverted image
        image_array = sitk.GetArrayFromImage(image)

        # Apply transpose the axis to the new positions and reverse direction if needed (-1)
        # Remember than sitk load the axis inverted in order (z=0,y=1,x=2), then we use 2 - x/y/z
        image_array = image_array.transpose((2-z_i,2-y_i,2-x_i))[::axis_idir[2],::axis_idir[1],::axis_idir[0]]

        # backup image info
        spacing, origin = image.GetSpacing(), image.GetOrigin()

        # set new direction and reorder origin and spacing accordingly
        x, y, z = np.where(axis_x)[0][0], np.where(axis_y)[0][0], np.where(axis_z)[0][0]
        new_direction = tuple(target_mat.reshape(-1)) #set the anatomical orientation
        new_spacing = (spacing[x], spacing[y], spacing[z])
        new_origin = (origin[x], origin[y], origin[z])
    else:
        raise RuntimeError("Function not implemented yet")


    #Create new image from transformed array
    image = sitk.GetImageFromArray(image_array)

    #Restore info
    image.SetOrigin(new_origin)
    image.SetSpacing(new_spacing)
    image.SetDirection(new_direction)
    return image
