from Generation import SliceGenerator
import argparse

parameters = {
    'dataset': "duke",
    'force_overwrite': False,
    'crop_breast': True,
    'center' : True,
    'serie_list': ["BH", "E2"],

    # Setup Processing options
    'input_format': ["pre", "post[1]", "post[-1]"],  # some e2 only have 3 post
    'post_idx': -1,  # process input error if sinlge post

    # slice parameters
    'slice_axis': 0,
    'percentage_slices': 0.0,

    # Subtype
    'class_labels': 'TN'
}

# Parser input arguments
parser = argparse.ArgumentParser(description='Gen. method.')
parser.add_argument('-d', '--dataset', type=str, default=parameters['dataset'], help='Dataset - [duke, brca, ispy1]')
parser.add_argument('-f', dest='force', type=bool, default=parameters['force_overwrite'], help='Force recompute dataset')
parser.add_argument('--crop', type=bool, default=parameters['crop_breast'], help='Crop breast')
parser.add_argument('--center', type=bool, default=parameters['center'], help='Center Image')
#parser.add_argument('--input', type=list, default=parameters['input_format'], help='learning rate')
parser.add_argument('-p', '--percentage', type=float, default=parameters['percentage_slices'], help='Percentage of tumor slices used')
parser.add_argument('-c', '--class', dest='class_type', type=str, default=parameters['class_labels'], help='Class used - [HR, TN, ER, PR, HER2, HER2+]')
# parser.add_argument('-n',  type=int, dest='num_executions', default=NUMBER_EXECUTIONS, help='number of execution')
args = parser.parse_args()

# set up training parameters
parameters['dataset'] = args.dataset
parameters['force_overwrite'] = args.force
parameters['crop_breast'] = args.crop
parameters['center'] = args.center
parameters['percentage_slices'] = args.percentage
parameters['class_labels'] = [args.class_type]



#SliceGenerator.SliceGeneration(options)
SliceGenerator.SliceGenerationMultiThread(parameters)