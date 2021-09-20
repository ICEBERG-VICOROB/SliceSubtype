import os

from fastai.imports import *
from fastai.torch_core import *
from fastai.learner import *

# Devide a number in two dimensions, with ceiling. EX: 4 -> 2x2
# use_round: In case of empty space ambigous case, force a rather square solution 7-> 3x3 or (4x2 or 2x4)
# x_wise: bigger length is x in case of non-square shape
# Ex:  3 - use_round: 2x2 otherwise (x_wise: 3x1 otherwise 1x3)
# EX: x_wise: 5 -> x: 3  y: 2   | not x_wise:  5 -> x: 2   y: 3
def squared_part(v, x_wise=True, use_round=True):
    if x_wise:
        y = int( round(math.sqrt(v)) if use_round else math.sqrt(v) )
        x = int( math.ceil(v / y) )
    else:
        x = int( round(math.sqrt(v)) if use_round else math.sqrt(v) )
        y = int( math.ceil(v / x) )
    return x, y

@patch
@delegates(subplots)
def plot_metrics(self: Recorder, nrows=None, ncols=None, figsize=None, **kwargs):
    return_fig = "return_fig" in kwargs and kwargs.pop("return_fig")
    metrics = np.stack(self.values)
    names = self.metric_names[1:-1]
    n = len(names) - 1
    if nrows is None and ncols is None:
        nrows = int(math.sqrt(n))
        ncols = int(np.ceil(n / nrows))
    elif nrows is None:
        nrows = int(np.ceil(n / ncols))
    elif ncols is None:
        ncols = int(np.ceil(n / nrows))
    figsize = figsize or (ncols * 6, nrows * 4)
    fig, axs = subplots(nrows, ncols, figsize=figsize, **kwargs)
    axs = [ax if i < n else ax.set_axis_off() for i, ax in enumerate(axs.flatten())][:n]
    for i, (name, ax) in enumerate(zip(names, [axs[0]] + axs)):
        ax.plot(metrics[:, i], color='#1f77b4' if i == 0 else '#ff7f0e', label='valid' if i > 0 else 'train')
        ax.set_title(name if i > 1 else 'losses')
        ax.legend(loc='best')
 #   plt.show()
    if return_fig:
        return fig


#https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
#https://www.kaggle.com/mamamot/fastai-v2-example

from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.metrics import roc_auc_score
def PlotRoc(preds, targets, labels=None):

    try:
        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
    except:
        pass

    num_classes = preds.shape[1]

    if labels is None:
        labels = list(range(num_classes))
    elif len(labels) != num_classes:
        raise RuntimeError("Wrong number of labels! Do not much num_classes in predictions")
    elif "micro" in labels or "macro" in labels:
        raise RuntimeError("Restricted labels names 'micro' and 'macro' cannot be used for labeling. Please, change the labels or use the labels=None parameter.")

    ctargets = np.array([ [x == i for i in range(num_classes)] for x in targets])
    #print(ctargets)
    #print(preds)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[labels[i]], tpr[labels[i]], _ = roc_curve(ctargets[:, i], preds[:, i])
        roc_auc[labels[i]] = auc(fpr[labels[i]], tpr[labels[i]])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve( ctargets.ravel(), preds.ravel() )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[labels[i]] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[labels[i]], tpr[labels[i]])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    fig = plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[labels[i]], tpr[labels[i]], color=color, #, lw=lw
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(labels[i], roc_auc[labels[i]]))

    plt.plot([0, 1], [0, 1], 'k--') #, lw=lw
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (multi-class)')
    plt.legend(loc="lower right")
    return fig, roc_auc
   # plt.show()


# TODO: Add image part
def PlotPred(df, learn, threshold=0.5, size_per_graph=4, save_image_folder=None, save_image=None ): # create_plot=True,

    val = df[df['is_valid'] == True]
    if isinstance(learn, tuple):
        preds, targs, valid_ds = learn
    else:
        preds, targs = learn.get_preds()
        valid_ds = learn.dls.valid_ds

    #for l, t in zip(val['labels'], targs):
    #    print(int(l),int(t))

    # Check that the information is correct
    if any(int(l) != int(t) for l, t in zip(val['labels'], targs)):
        raise RuntimeError("Unexpected behavior, df table and predictions labels do not match!")

    res_dict = { "targets": [], "preds": [], "fnames": [], "PILs": [], "names":[] }
    for f, l, n, t, p, data in zip(val['fname'], val['labels'], val['name'], targs, preds, valid_ds):
        res_dict["targets"].append(int(t))
        res_dict["preds"].append(p.tolist())
        res_dict["fnames"].append(f)
        res_dict["PILs"].append(data[0])
        res_dict["names"].append(n)
        #print(f,l,t,p,data)

    if save_image_folder is not None and not os.path.exists(save_image_folder):
        os.mkdir(save_image_folder)
        res_dict_names = {}

        for i in range(len(targs)):
            if res_dict["names"][i] not in res_dict_names:
                res_dict_names[res_dict["names"][i]] = []
            res_dict_names[res_dict["names"][i]].append(i)

        for k, items in res_dict_names.items():
            x_axis, y_axis = squared_part(len(items))
            print(x_axis, y_axis)
            fig, axs = plt.subplots(y_axis, x_axis)
            for i, idx in enumerate(items):
                px = i % x_axis
                py = i // x_axis
                t = res_dict["targets"][idx]
                p = 1 if (res_dict["preds"][idx][1] > threshold) else 0
                if y_axis > 1:
                    axs[py, px].set_title('pred {} ({})\n{}'.format(p, t, res_dict["fnames"][idx]),
                                          color=('green' if t == p else 'red'))
                    axs[py, px].imshow(res_dict["PILs"][idx])
                elif x_axis > 1:
                    axs[px].set_title('pred {} ({})\n{}'.format(p, t, res_dict["fnames"][idx]),
                                          color=('green' if t == p else 'red'))
                    axs[px].imshow(res_dict["PILs"][idx])
                else:
                    axs.set_title('pred {} ({})\n{}'.format(p, t, res_dict["fnames"][idx]),
                                      color=('green' if t == p else 'red'))
                    axs.imshow(res_dict["PILs"][idx])

            fig.set_size_inches(size_per_graph * x_axis, size_per_graph * y_axis)
            fig.savefig(os.path.join(save_image_folder, "{}.jpg".format(k)))
            plt.close(fig)


    if save_image is not None and not os.path.exists(save_image):
        x_axis, y_axis = squared_part(len(targs))
        fig1, axs = plt.subplots(y_axis, x_axis)
        for i in range(len(targs)):
            px = i % x_axis
            py = i // x_axis
            t = res_dict["targets"][i]
            p = 1 if (res_dict["preds"][i][1] > threshold) else 0
            axs[py, px].set_title('pred {} ({})\n{}'.format(p,t,res_dict["fnames"][i]), color=('green' if t==p else 'red'))
            axs[py ,px].imshow(res_dict["PILs"][i])
        fig1.set_size_inches(size_per_graph * x_axis, size_per_graph * y_axis)
        image_filename = os.path.splitext(save_image)[0] + ".jpg" #remove extension and add jpg
        fig1.savefig(image_filename)





    return {  "targets": res_dict["targets"] , "preds": res_dict["preds"] , "fnames": res_dict["fnames"], "names": res_dict["names"]  }




def PlotConfusionMat(y_preds, targs, labels, normalize=False, title='Confusion matrix', cmap="Blues", norm_dec=2, plot_txt=True, **kwargs ):

    x = np.arange(0, len(labels))
    d, t = y_preds, targs
    cm = ((d == x[:, None]) & (t == x[:, None, None])).sum(2)
    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure(**kwargs)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels, rotation=0)

    if plot_txt:
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            coeff = f'{cm[i, j]:.{norm_dec}f}' if normalize else f'{cm[i, j]}'
            plt.text(j, i, coeff, horizontalalignment="center", verticalalignment="center", color="white" if cm[i, j] > thresh else "black")

    ax = fig.gca()
    ax.set_ylim(len(labels)-.5,-.5)

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.grid(False)

    return fig


#interpre.preds, interpre.targs

#step, max 4 decimals
# Get and plot different metrics by threshold value
def ResumeThresholds(preds, targs, labels=None, plot=1, step=0.1, linewidth=2, plot_optimal_th = True, size_per_graph=5, return_fig=False):

    if labels is None:
        labels = list(range(preds.shape[1]))
    elif len(labels) != preds.shape[1]:
        raise RuntimeError("Wrong number of labels! Do not much num_classes in predictions")

    best_threshold = {}
    threshold_list = {}
    for l_idx, label in enumerate(labels):
        threshold_list[label] = {}
        for th in np.arange(0, 1+step, step):
            th = round(th,4) #step, max for decimals!
            tp = 0
            tn = 0
            for p, t in zip(preds, targs):
                if t == l_idx:
                    tp += (1 if p[l_idx] >= th else 0)
                else:
                    tn += (1 if p[l_idx] < th else 0)
            accuracy = (tp + tn) / len(targs)

            recall = tp / sum(targs == l_idx).item() #as it is tensor, extract the item (TODO: check is vector)
            specificity = tn / sum(targs != l_idx).item()
            fp = sum(targs != l_idx).item() - tn
            #print(l_idx, sum(targs == l_idx), float(sum(targs == l_idx)), sum(targs == l_idx).item(), tp, fp)
            if tp + fp == 0:
                precision = 1.0 #when 0 cases are predicted as positive (right or wrong), then 1 or nan?
            else:
                precision = tp / (tp + fp)
            #print(precision)
            if precision + recall == 0:
                f1 = math.nan #correct? https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            threshold_list[label][th] = {"accuracy": accuracy, "recall": recall, "specificity": specificity,
                                  "precision": precision, "f1": f1}

        #find best threshold
        best_threshold[label] = -1
        best_min_dist = 100000
        for th, results in threshold_list[label].items():
            #minimum distance between metrics is the norm of their values minus their mean
            min_dist = np.linalg.norm(np.array(list(results.values())) - np.average(list(results.values())))
            #print(label, th, min_dist)
            if min_dist < best_min_dist:
                best_min_dist = min_dist
                best_threshold[label] = th
        if best_threshold[label] == -1:
            raise RuntimeError("Unexpected error, labels or metrics maybe missing")


    if plot == 1 or plot == 3:
        x_axis, y_axis = squared_part(len(labels), x_wise=True)
        #print(x_axis, y_axis)
        fig1, axs = plt.subplots(y_axis, x_axis)
        for l_idx, label in enumerate(labels):
            px = l_idx % x_axis
            py = l_idx // x_axis
            if y_axis > 1:
                axs[py, px].set_title('Label {}'.format(label))
            else:
                axs[px].set_title('Label {}'.format(label))
            results_dict = {}
            for th, results in threshold_list[label].items():
                for key, value in results.items():
                    if key not in results_dict:
                        results_dict[key] = []
                    results_dict[key].append(value)
            th_list = list( threshold_list[label].keys())
            for key, values in results_dict.items():
                if y_axis > 1:
                    axs[py, px].plot(th_list, values, label=key, linewidth=linewidth)
                else:
                    axs[px].plot(th_list, values, label=key, linewidth=linewidth)
            if y_axis > 1:
                if plot_optimal_th:
                    axs[py, px].plot([best_threshold[label],best_threshold[label]], [0,1], label="Optimal th",
                                     linestyle=':', linewidth=linewidth, color='black')
                axs[py, px].legend(loc="lower right")
               # axs[py, px].xticks(th_list)
            else:
                if plot_optimal_th:
                    axs[px].plot([best_threshold[label], best_threshold[label]], [0, 1], label="Optimal th",
                                     linestyle=':', linewidth=linewidth, color='black')
                axs[px].legend(loc="lower right")
#                axs[px].xticks(th_list)
        fig1.set_size_inches(size_per_graph*x_axis, size_per_graph*y_axis)

    if plot == 2 or plot == 3:
        results_dict = {}
        for l_idx, label in enumerate(labels):
            for th, results in threshold_list[label].items():
                for key, value in results.items():
                    if key not in results_dict:
                        results_dict[key] = {}
                    if label not in results_dict[key]:
                        results_dict[key][label] = []
                    results_dict[key][label].append(value)

        th_list = list(threshold_list[label].keys())
        x_axis, y_axis = squared_part(len(results_dict), x_wise=True)
        fig2, axs = plt.subplots(y_axis, x_axis)
        for i, (key, results) in enumerate(results_dict.items()):
            px = i % x_axis
            py = i // x_axis
            if y_axis > 1:
                axs[py, px].set_title('Metric {}'.format(key))
            else:
                axs[px].set_title('Metric {}'.format(key))

            for label, values in results.items():
                if y_axis > 1:
                    if plot_optimal_th:
                        axs[py, px].plot([best_threshold[label], best_threshold[label]], [0, 1],
                                         label="Optimal th ({})".format(label),
                                         linestyle=':', linewidth=linewidth)
                    axs[py, px].plot(th_list, values, label=label, linewidth=linewidth)
                else:
                    if plot_optimal_th:
                        axs[px].plot([best_threshold[label], best_threshold[label]], [0, 1],
                                     label="Optimal th ({})".format(label),
                                     linestyle=':', linewidth=linewidth)
                    axs[px].plot(th_list, values, label=label, linewidth=linewidth)


            if y_axis > 1:
                axs[py, px].legend(loc="lower right")
                #axs[py, px].xticks(th_list)
            else:
                axs[px].legend(loc="lower right")
               # axs[px].xticks(th_list)
        fig2.set_size_inches(size_per_graph * x_axis, size_per_graph * y_axis)


    if return_fig and plot == 1:
        return threshold_list, best_threshold, fig1
    elif return_fig and plot == 2:
        return threshold_list, best_threshold, fig2
    elif return_fig and plot == 3:
        return threshold_list, best_threshold, fig1, fig2
    else:
        return threshold_list, best_threshold