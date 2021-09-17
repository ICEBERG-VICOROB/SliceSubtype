import matplotlib

matplotlib.use('TkAgg')
from fastai.vision.all import *  # important
import random
from datetime import datetime
from shutil import copyfile
from . PlottingMetrics  import *
import seaborn as sns
import argparse
import os

import json


def ShowParameters(dict_param):
    print("Params:")
    for k, param in dict_param.items():
        print("{}: {}".format(k, param))
    print("")


def TempCopySourceCode(temp_filename=".filecopytemp"):
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        if os.path.exists(temp_filename):
            raise RuntimeError("File should not exists")
    copyfile(__file__, temp_filename)


def Status(v, total, text="", bars=20):
    bar = ("{:<" + str(bars) + "}").format("#" * int(bars * v / total))
    print("Status: [{}] {:>3}%  ({:>6}/{:<6})   {}".format(bar, int(100 * v / total), v, total, text))



def GetEfficientNet(dataloader_channels, b = 4, pretrained = True, in_channels = 3):

    assert (b >= 0 and b <= 8)

    efficient_net_name = "efficientnet-b{}".format(b)

    # https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html
    from efficientnet_pytorch import EfficientNet
    if pretrained:
        model = EfficientNet.from_pretrained(efficient_net_name, in_channels=in_channels, advprop=True)
    else:
        model = EfficientNet.from_name(efficient_net_name, in_channels=in_channels)

    # model = EfficientNet.from_name("efficientnet-b7")
    # model = EfficientNet.from_pretrained("efficientnet-b8", advprop=True) # weights run to NaN

    if b == 0 or b == 1:
        model._fc = nn.Linear(1280, dataloader_channels) # the last layer... # works for b0,b1
    elif b == 3:
        model._fc = nn.Linear(1536, dataloader_channels)  # the last layer... B3
    elif b == 4:
        model._fc = nn.Linear(1792, dataloader_channels)  # the last layer... B4
    elif b == 5:
        model._fc = nn.Linear(2048, dataloader_channels) # the last layer... B5
    elif b == 6:
        model._fc = nn.Linear(2304, dataloader_channels) # the last layer... B6
    elif b == 7:
        model._fc = nn.Linear(2560, dataloader_channels) # the last layer... B7
    elif b == 8:
        model._fc = nn.Linear(2816, dataloader_channels) # the last layer... B8

    return model





def BalanceImages(df, balance_field="labels", non_valid_only=True):
    masked_df = df[df['is_valid'] == False] if non_valid_only else df
    # Group gets all cases the same togethers, so the same cases for the same name are joined in each group therefore we zip and concatanate all blocks together
    # If the slice from the same image are concataneted then some images have double slices and others not, then this create bigger valdation than train (becpuse we devide by image num no slice num)
    # list(itertools.chain.from_iterable(zip(*v.groupby("name").groups.values())))
    list_cases = {k: list(v.index) for k, v in masked_df.groupby(balance_field)}
    # print(list_cases)
    max_list = max([len(item) for k, item in list_cases.items()])
    for k, v in list_cases.items():
        diff = max_list - len(v)
        # print(len(v), max_list, diff)
        new_l = []
        if diff > 0:
            for i in range(0, diff):  # Copy a random element each time
                if len(new_l) == 0:
                    new_l = v.copy()
                    random.shuffle(new_l)
                df = df.append(df.iloc[new_l.pop()].to_dict(), ignore_index=True)

    return df



def Train(dataframe, n_epochs, batch_size, lr, balance_training=True, evaluate = True, output_folder = None,
          df_label='labels', df_filename='fname', df_group_field='name', files_path=""):

    output = {}

    output["input_figure"], ax_input = plt.subplots(1 , 3, figsize=(12, 4) )

    # sns.countplot(y='Pam50.Call',data=df, hue="Pam50.Call",ax=ax[0])
    ax_input[0].set_title("Validation division for group")
    sns.countplot(y="is_valid", data=dataframe.groupby(df_group_field).head(1), hue=df_label,
                  ax=ax_input[0] )
    # ax[1].set_title("Image division")
    ax_input[1].set_title("Validation division for images")
    sns.countplot(y="is_valid", data=dataframe, hue=df_label, ax=ax_input[1])

    if balance_training:
        print("Balancing training data...")
        dataframe = BalanceImages(dataframe)

        ax_input[2].set_title("Validation division for images (Balanced)")
        sns.countplot(y="is_valid", data=dataframe, hue=df_label, ax=ax_input[2])

    if output_folder:
        if os.path.exists(output_folder):
            print("Error: Output folder already exists!")
        else:
            os.mkdir(output_folder)
            output["input_figure"].savefig(os.path.join(output_folder, "input.png"))

    # Create Datablock
    images = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       splitter=ColSplitter(),  # Use is_valid column
                       get_x=ColReader(df_filename, pref=Path(files_path)),
                       get_y=ColReader(df_label),
                       item_tfms=Resize(224),
                       batch_tfms=
                       [Normalize.from_stats(*imagenet_stats), *aug_transforms(max_warp=0, min_scale=0.75)])
                        # *aug_transforms(size=224, min_scale=0.75),/worse? ->*aug_transforms()

    # Create dataloader
    dls = images.dataloaders(dataframe, bs=batch_size)  # important default bs(64) is very big for few images, so we set 5

    # Define network model
    model = GetEfficientNet(dls.c, b=4)

    #Define Learner
    learn = Learner(dls, model, metrics=[error_rate])  # .to_fp16() #, roc_auc_bin  roc_auc = RocAuc()  # https://forums.fast.ai/t/problem-with-f1scoremulti-metric/63721

    # Train
    print("Training... (LR: {})".format(lr))
    learn.fit_one_cycle(n_epochs, lr)

    output["plot_metrics"] = learn.recorder.plot_metrics(return_fig=True)
    #output["plot_lr"] = learn.recorder.plot_lr(return_fig=True)
    #output["plot_losses"] = learn.recorder.plot_losses(return_fig=True)
    output["learner"] = learn

    # Evaluate
    if evaluate:
        # Create classification iterpreter
        interpre = ClassificationInterpretation.from_learner(learn, dl=dls.valid)

        # Save results
        output["evaluate"] = {}
        output["evaluate"]["dataset"] = dataframe[dataframe['is_valid'] == True] #learn.dls.valid_ds
        output["evaluate"]["preds"] = interpre.preds.detach().cpu().numpy()
        output["evaluate"]["targs"] = interpre.targs.detach().cpu().numpy()
        output["evaluate"]["labels"] = list(interpre.vocab.o2i.values())

        output["evaluate"]["roc"], output["evaluate"]["roc_auc"] = PlotRoc(output["evaluate"]["preds"], output["evaluate"]["targs"])


    if output_folder:
        output["plot_metrics"].savefig(os.path.join(output_folder, "metrics.png"))
        #output["plot_lr"].savefig(os.path.join(output_folder, "lr.png"))
        #output["plot_losses"].savefig(os.path.join(output_folder, "losses.png"))
        if evaluate:
            output["evaluate"]["roc"].savefig(os.path.join(output_folder, "roc.png"))

    return output




