import matplotlib
from Training.TrainKFold import *

matplotlib.use('TkAgg')
from Training.PlottingMetrics import *
import argparse
import os

# TODO: Add tensorboard support
# from fastai.callbacks.tensorboard import *
# tboard_path = Path('**path-to-your-log**')
# learn.callback_fns.append(partial(LearnerTensorboardWriter, base_dir=tboard_path, name='Smthng'))

# Check: Learning rate
# https://blog.dataiku.com/the-learning-rate-finder-technique-how-reliable-is-it
# https://forums.fast.ai/t/automated-learning-rate-suggester/44199
# https://fastai1.fast.ai/callbacks.lr_finder.html
# https://forums.fast.ai/t/new-fastai-lr-find-recorder-plot-equivalent/76566/2
# https://walkwithfastai.com/lr_finder


# Default parameters
parameters = {
    # basic parameters
    "training_data_folder": "duke_TN_000",
    "output_path": "results",
    "number_executions": 1,

    # Training parameters
    "num_epochs": 40,
    "batch_size": 20,
    "learning_rate": 3e-3,
    "balance_training_data": True,
    "check_leraning_rate": False,
    "k_fold": 4,

    # Others
    "show_input_samples": 0,
    "show_output": False,
    "save_per_k_res": True,
    "copy_source_code": True,
}

# Parser input arguments
parser = argparse.ArgumentParser(description='Train method.')
parser.add_argument('-f', '--folder', type=str, default=parameters["training_data_folder"], help='data folder')
parser.add_argument('-o', '--output', type=str, default=parameters["output_path"], help='results main path')
parser.add_argument('-e', type=int, dest='epochs', default=parameters["num_epochs"], help='num epochs')
parser.add_argument('-b', type=int, dest='batch_size', default=parameters["batch_size"], help='batchsize')
parser.add_argument('-l', type=float, dest='learning_rate', default=parameters["learning_rate"], help='learning rate')
parser.add_argument('-k', type=int, dest='k_fold', default=parameters["k_fold"], help='number of K-FOLDS')
# parser.add_argument('-n',  type=int, dest='num_executions', default=NUMBER_EXECUTIONS, help='number of execution')
args = parser.parse_args()

# set up training parameters
parameters["training_data_folder"] = args.folder
parameters["output_path"] = args.output
parameters["num_epochs"] = args.epochs
parameters["batch_size"] = args.batch_size
parameters["learning_rate"] = args.learning_rate
parameters["k_fold"] = args.k_fold


# todo:
# class imbalance: https://forums.fast.ai/t/handling-class-imbalance-in-deep-learning-models/33054
#                   https://forums.fast.ai/t/highly-imbalanced-data/37001/11
#                   https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109

# Load csv with slice names for training
print("Loading CSV...")
path = parameters["training_data_folder"]
df = pd.read_csv(os.path.join(path,'train.csv'))  # , index_col=0

# Show loading info
print("Input:", path)
print("Num groups:", len(df.groupby("name")))
print("Num total slices:", len(df))
print(df.head())
print("")


if not os.path.exists(parameters["output_path"]):
    os.mkdir(parameters["output_path"])

# Set output folder
input_folder_basename = os.path.basename(parameters["training_data_folder"])[:30]
current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
output_folder = os.path.join(parameters["output_path"], "{}-{}-k{}-e{}-b{}-l{:0.4f}".format(current_date,
                                                                                            input_folder_basename,
                                                                                            parameters["k_fold"],
                                                                                            parameters["num_epochs"],
                                                                                            parameters["batch_size"],
                                                                                            parameters["learning_rate"]))
print("Result path:", output_folder)


# Train K-fold
output = TrainKFold(df, k_folds=parameters["k_fold"],
                    n_epochs=parameters["num_epochs"],
                    batch_size=parameters["batch_size"],
                    lr=parameters["learning_rate"],
                    output_folder=output_folder,
                    files_path=path)



