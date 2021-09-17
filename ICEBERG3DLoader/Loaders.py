from ICEBERG3DLoader import BRCALoader, ISPY1Loader, DukeLoader
import os
import yaml

### DEFAULT OPTIONS ####
default_options = {'dataset_folder': "./dataset_folder",
                   'les_folder': "./les_folder",
                   'data_gt': "", #join les_folder and data_gt to ./gt file or folder?
                   'data_folder': "./formated_dataset",
                   'crop_breast': False,
                   'orientation': None,
                   'center': False,
                   'bias_field_n4': False,
                   'save_metadata': False,
                   'serie_list': None,
                   'post_subset': None,
                   'time_list': None,
                   'time_index': None, #t1, t2, t3, t4 oltherwise use first one avaliable
                   'force_overwrite': False,
                   'dataset_name': None}
###############################33


# Universal Loader
def LoadDataset(options_dict = {}, file_list = None, load_only=False):

    if 'dataset' not in options_dict:
        raise RuntimeError("Parameter 'dataset' missing. Please specify the parameter or load the dataset with the correct loader")

    dataset = options_dict['dataset'].lower()

    supported_datasets = ['bcra', 'ispy1', 'duke']

    if dataset == "brca":
        return LoadBRCADataset(options_dict, file_list, load_only)
    elif dataset == 'ispy1':
        return LoadISPY1Dataset(options_dict, file_list, load_only)
    elif dataset == 'duke':
        return LoadDukeDataset(options_dict, file_list, load_only)
    else:
        raise RuntimeError("Parameter 'dataset' unknown. Range({})".format(supported_datasets))




#'data_gt' instead of les_folder
def LoadBRCADataset(options_dict = {}, file_list = None, load_only=False):

    #Auto-complete the options with default value
    options = default_options.copy() #.update(options_dict)
    options.update(options_dict)

    # Load data if found
    if file_list is None or not os.path.exists(file_list):

        # Load/Format Dataset
        if not load_only:
            dataset = BRCALoader.AutoFormatDataset(options['dataset_folder'], options['data_gt'], options['data_folder'],
                                        options['crop_breast'], options['orientation'], options['center'],
                                        options['bias_field_n4'],
                                        options['save_metadata'], options['serie_list'], options['post_subset'],
                                        options['force_overwrite'], options['dataset_name'])
        else:
            if not isinstance(options['dataset_name'], str):
                dataset_name = BRCALoader.GetDatasetName(options['crop_breast'], options['orientation'],
                                              options['center'], options['bias_field_n4'], options['post_subset'])
            formated_folder = os.path.join(options['data_folder'], dataset_name)
            dataset = BRCALoader.LoadFromattedDataset(formated_folder, options['serie_list'])


        pre, post, labels = BRCALoader.FormatFilesLists(dataset, post_idx=options['post_idx'], shuffle=True, maximum_files=None)

        # Save files list if activated
        if file_list is not None:
            SaveFile(file_list, { "pre": pre, "post": post, "labels": labels} )
            print("File list <{}> saved!".format(options['file_list']))

    else:
        # Load file list
        loaded_file = LoadFile(file_list)
        if "pre" not in loaded_file or "post" not in loaded_file or "labels" not in loaded_file:
            raise RuntimeError("File list loading error. Unknown format.")

        pre, post, labels = loaded_file["pre"], loaded_file["post"], loaded_file["labels"]

        # Check all files exists
        if any(not os.path.exists(f) for f in pre + post + labels):
            raise RuntimeError("File list loading error. Indicated file not found.")

        print("File list <{}> loaded successfuly!".format(options['file_list']))

    return pre, post, labels



def LoadISPY1Dataset(options_dict, file_list = None, load_only=False):

    #Auto-complete the options with default value
    options = default_options.copy() #.update(options_dict)
    options.update(options_dict)

    # Load data if found
    if file_list is None or not os.path.exists(file_list):

        # Load/Format Dataset
        if not load_only:
            dataset = ISPY1Loader.AutoFormatDataset(options['dataset_folder'], options['data_folder'],
                                        options['crop_breast'], options['orientation'], options['center'],
                                        options['bias_field_n4'],
                                        options['save_metadata'], options['time_list'], options['post_subset'],
                                        options['force_overwrite'], options['dataset_name'])
        else:
            if not isinstance(options['dataset_name'], str):
                dataset_name = ISPY1Loader.GetDatasetName(options['crop_breast'], options['orientation'],
                                              options['center'], options['bias_field_n4'], options['post_subset'])
            formated_folder = os.path.join(options['data_folder'], dataset_name)
            dataset = ISPY1Loader.LoadFromattedDataset(formated_folder, options['time_list'])


        pre, post, labels = ISPY1Loader.FormatFilesLists(dataset, post_idx=options['post_idx'], shuffle=True, maximum_files=None, time_item=options['time_index'])

        # Save files list if activated
        if file_list is not None:
            SaveFile(file_list, { "pre": pre, "post": post, "labels": labels} )
            print("File list <{}> saved!".format(options['file_list']))

    else:
        # Load file list
        loaded_file = LoadFile(file_list)
        if "pre" not in loaded_file or "post" not in loaded_file or "labels" not in loaded_file:
            raise RuntimeError("File list loading error. Unknown format.")

        pre, post, labels = loaded_file["pre"], loaded_file["post"], loaded_file["labels"]

        # Check all files exists
        if any(not os.path.exists(f) for f in pre + post + labels):
            raise RuntimeError("File list loading error. Indicated file not found.")

        print("File list <{}> loaded successfuly!".format(options['file_list']))

    return pre, post, labels


def LoadDukeDataset(options_dict, file_list = None, load_only=False):

    #Auto-complete the options with default value
    options = default_options.copy() #.update(options_dict)
    options.update(options_dict)

    # Load data if found
    if file_list is None or not os.path.exists(file_list):

        # Load/Format Dataset
        if not load_only:
            dataset = DukeLoader.AutoFormatDataset(options['dataset_folder'], options['data_gt'], options['data_folder'],
                                        options['crop_breast'], options['orientation'], options['center'],
                                        options['bias_field_n4'],
                                        options['save_metadata'], options['post_subset'],
                                        options['force_overwrite'], options['dataset_name'])
        else:
            if not isinstance(options['dataset_name'], str):
                dataset_name = ISPY1Loader.GetDatasetName(options['crop_breast'], options['orientation'],
                                              options['center'], options['bias_field_n4'], options['post_subset'])
            formated_folder = os.path.join(options['data_folder'], dataset_name)
            dataset = DukeLoader.LoadFromattedDataset(formated_folder)


        pre, post, labels = DukeLoader.FormatFilesLists(dataset, post_idx=options['post_idx'], shuffle=True, maximum_files=None)

        # Save files list if activated
        if file_list is not None:
            SaveFile(file_list, { "pre": pre, "post": post, "labels": labels} )
            print("File list <{}> saved!".format(options['file_list']))

    else:
        # Load file list
        loaded_file = LoadFile(file_list)
        if "pre" not in loaded_file or "post" not in loaded_file or "labels" not in loaded_file:
            raise RuntimeError("File list loading error. Unknown format.")

        pre, post, labels = loaded_file["pre"], loaded_file["post"], loaded_file["labels"]

        # Check all files exists
        if any(not os.path.exists(f) for f in pre + post + labels):
            raise RuntimeError("File list loading error. Indicated file not found.")

        print("File list <{}> loaded successfuly!".format(options['file_list']))

    return pre, post, labels

#Save and load files functions

def SaveFile(file, dict_data):
    with open(file, 'w', encoding='utf8') as outfile:
        yaml.dump(dict_data, outfile, default_flow_style=False, allow_unicode=True)

def LoadFile(file):
    with open(file, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded
