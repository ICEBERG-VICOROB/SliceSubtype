# Datasets configuration
datasets = {
    "duke": {   'dataset_folder': "/data/datasets/MRI_Duke/manifest-1607053360376/Duke-Breast-Cancer-MRI",
                'data_folder': "/data/datasets/MRI_Duke/Duke-Formated", #output formated dataset folder
                'data_gt': "/data/datasets/MRI_Duke/Annotation_Boxes.csv",
                'subtype_csv': "/data/datasets/MRI_Duke/Clinical_and_Other_Features.csv"
    },
    "ispy1": {  'dataset_folder': "/data/datasets/ISPY1_2a/ISPY1",
                'data_folder': "/data/datasets/ISPY1_2a/ISPY1-Formated", #output formated dataset folder
                'data_gt': "",
                'subtype_csv': '/data/datasets/ISPY1_2a/TCIA_PAtient_Clinical_Subset.csv'
    },
    "brca": {   'dataset_folder': "/data/datasets/TCGA-BRCA",
                'data_folder': "/data/datasets/tcga2/TCGA-BRCA-Formated", #output formated dataset folder
                'data_gt': "/data/datasets/tcga2/TCGA_Segmented_Lesions_UofC",
                'subtype_csv': '/data/datasets/tcga2/nationwidechildrens.org_clinical_patient_brca.csv'
    }
}

default_options = {
    'dataset': "duke",  # Options: duke, ispy1, brca
    'force_overwrite': False,
    'crop_breast': False,
    'orientation': None,
    'center': False,
    'bias_field_n4': False,
    'save_metadata': False,
    'serie_list': None,
    'post_subset': None,
    'time_list': None,
    'time_index': None, #t1, t2, t3, t4 oltherwise use first one avaliable
    'dataset_name': None,
    'file_list': None,
    'input_format': ["pre", "post[0]", "post[1]"],  # some e2 only have 3 post
    'post_idx': -1,  # process input error if sinlge post
    'slice_axis': 0,
    'percentage_slices': 1.0,
    'class_labels': "HR",
    'slice_folder': None
}
