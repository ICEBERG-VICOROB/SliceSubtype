import pandas as pd



def SubtypeData(dataset, filename, pos=1, neg=0, unclear="unknown"):

    supported_datasets = ['bcra', 'ispy1', 'duke']

    if dataset == "brca":
        return BRCASubtypeData(filename, pos=1, neg=0, unclear="unknown")
    elif dataset == 'ispy1':
        return ISPY1SubtypeData(filename, pos=1, neg=0, unclear="unknown")
    elif dataset == 'duke':
        return DukeSubtypeData(filename, pos=1, neg=0, unclear="unknown")
    else:
        raise RuntimeError("Parameter dataset unknown. Range({})".format(supported_datasets))


# Load Duke Subtype Data
# filename: csv or xls file with ER, PR and HER2 headers with (1,0) data
# pos, neg: output value of the subtype
# unclear: output unclear/unknown value of subtype
def DukeSubtypeData(filename, pos=1, neg=0, unclear="unknown"): #unclear can be None to be disabled

    if neg == pos or unclear == pos:
        raise RuntimeError("Labels values cannot be equal!")

    print("Reading subtype file...")
    if filename.endswith(".csv"):
        df = pd.read_csv(filename, header=[0, 1, 2], index_col=0)
    elif filename.endswith(".xls") or filename.endswith(".xlsx"):
        df = pd.read_excel(filename, header=[0, 1, 2], index_col=0)
    else:
        raise RuntimeError("Unknown file format. File must use the (csv/xls/xlsx) extension.")

    #Filter headers, only second one is used
    df.columns = [h[1] for h in df.columns]

    headers_list = ['ER', 'PR', 'HER2', 'Mol Subtype']

    print("Formating subtype...")

    #Filter subtypes columns
    df = df[headers_list]

    #Rename unclear
    if unclear is not None:
        df = RenameUnclearList(df, headers_list, 1, 0, unclear)

    #Add missing items HR, HER2+ and TN
    df = AddHRType(df, 'ER', 'PR', 'HR', 1, 0, unclear) #HR
    df = AddHer2EnrichedType(df, 'HR', 'HER2', 'HER2+', 1, 0, unclear)  # HER2+ (Enriched)
    df = AddTripleNegativeType(df, 'HR', 'HER2', 'TN', 1, 0, unclear)  # TN

    if pos != 1 or neg != 0:
        print("Replacing [1,0] -> [{},{}] ...".format(pos, neg))
        df = df.replace( {1: pos, 0: neg} )

    headers_list += ["HR", "HER2+", "TN"]
    print("Headers:", headers_list)

    df.convert_dtypes()

    return df

# Load BRCA Subtype Data (Only immunohistochemistry*)
# filename: csv or xls file with er_status_by_ihc, pr_status_by_ihc and her2_status_by_ihc headers with (Positive,Negative) data
# pos, neg: output value of the subtype
# unclear: output unclear/unknown value of subtype
def BRCASubtypeData(filename, pos=1, neg=0, unclear="unknown"): #unclear can be None to be disabled

    if neg == pos or unclear == pos:
        raise RuntimeError("Labels values cannot be equal!")

    print("Reading subtype file...")
    if filename.endswith(".csv"):
        df = pd.read_csv(filename, header=[0, 1, 2], index_col=1)
    elif filename.endswith(".xls") or filename.endswith(".xlsx"):
        df = pd.read_excel(filename, header=[0, 1, 2], index_col=1)
    else:
        raise RuntimeError("Unknown file format. File must use the (csv/xls/xlsx) extension.")


    # Filter headers
    # Currently all three headers are used as comparison
    replace_header_dict = {
        ("er_status_by_ihc", "breast_carcinoma_estrogen_receptor_status", "CDE_ID:2957359"): "ER",
        ("pr_status_by_ihc", "breast_carcinoma_progesterone_receptor_status", "CDE_ID:2957357"): "PR",
        ("her2_status_by_ihc", "lab_proc_her2_neu_immunohistochemistry_receptor_status", "CDE_ID:2957563"): "HER2"
    }
    df.columns = [replace_header_dict[h] if h in replace_header_dict else h for h in df.columns]

    headers_list = ['ER', 'PR', 'HER2']

    print("Formating subtype...")

    #Filter subtypes columns
    df = df[headers_list]

    #Rename unclear
    if unclear is not None:
        df = RenameUnclearList(df, headers_list, "Positive", "Negative", unclear)

    #Add missing items HR, HER2+ and TN
    df = AddHRType(df, 'ER', 'PR', 'HR', "Positive", "Negative", unclear) #HR
    df = AddHer2EnrichedType(df, 'HR', 'HER2', 'HER2+', "Positive", "Negative", unclear)  # HER2+ (Enriched)
    df = AddTripleNegativeType(df, 'HR', 'HER2', 'TN', "Positive", "Negative", unclear)  # TN

    if pos != "Positive" or neg != "Negative":
        print("Replacing [1,0] -> [{},{}] ...".format(pos, neg))
        df = df.replace( {"Positive": pos, "Negative": neg} )

    headers_list += ["HR", "HER2+", "TN"]
    print("Headers:", headers_list)

    df.convert_dtypes()

    return df


# Load BRCA Subtype Data
# filename: csv or xls file with ERpos, PgRpos and HR Pos, Her2MostPos headers with (1,0) data
# pos, neg: output value of the subtype
# unclear: output unclear/unknown value of subtype
def ISPY1SubtypeData(filename, pos=1, neg=0, unclear="unknown"): #unclear can be None to be disabled

    if neg == pos or unclear == pos:
        raise RuntimeError("Labels values cannot be equal!")

    print("Reading subtype file...")
    if filename.endswith(".csv"):
        df = pd.read_csv(filename, index_col=0)
    elif filename.endswith(".xls") or filename.endswith(".xlsx"):
        df = pd.read_excel(filename, index_col=0)
    else:
        raise RuntimeError("Unknown file format. File must use the (csv/xls/xlsx) extension.")

    #Add dataset name in front of the patient number for indexing following the folders names
    df.index = ["ISPY1_{}".format(i) for i in df.index.to_list()]

    # Filter headers
    replace_header_dict = {
        "ERpos": "ER",
        "PgRpos": "PR",
        "HR Pos": "HR",
        "Her2MostPos": "HER2"
    }
    df.columns = [replace_header_dict[h] if h in replace_header_dict else h for h in df.columns]

    headers_list = ['ER', 'PR', "HR", 'HER2']

    print("Formating subtype...")

    #Filter subtypes columns
    df = df[headers_list]

    #Rename unclear
    if unclear is not None:
        df = RenameUnclearList(df, headers_list, 1, 0, unclear)

    #Add missing items HR, HER2+ and TN
    df = AddHer2EnrichedType(df, 'HR', 'HER2', 'HER2+', 1, 0, unclear)  # HER2+ (Enriched)
    df = AddTripleNegativeType(df, 'HR', 'HER2', 'TN', 1, 0, unclear)  # TN


    if pos != 1 or neg != 0:
        print("Replacing [1,0] -> [{},{}] ...".format(pos, neg))
        df = df.replace( {1: pos, 0: neg} )

    headers_list += ["HR", "HER2+", "TN"]
    print("Headers:", headers_list)

    df.convert_dtypes()

    return df




# Function to filter all entries with an unclear label for a specific header
def FilterUnclearLabel(df, header, unclear_label = "unknown"):
    return df[df[header] != unclear_label]




# Functions to compute missing subtypes

def AddHRType(df, er_header, pr_header, hr_header="HR", pos = "Positive", neg = "Negative", unclear_label = "unknown"):
    new_column = [row[1][er_header] if row[1][pr_header] == neg else pos for row in df.iterrows()]  # (row_index, set of columns)
    df[hr_header] = new_column
    if unclear_label is not None:
        unclear_mask = GetUnclearList(df, [er_header, pr_header], pos, neg)
        df = RenameUnclearMask(df, hr_header, unclear_mask, unclear_label)
    return df

def AddHer2EnrichedType(df, hr_header, her2_header, her2enriched_header = "HER2+", pos = "Positive", neg = "Negative", unclear_label = "unknown"):
    new_column = [ pos if row[1][hr_header] == neg and row[1][her2_header] == pos else neg for row in df.iterrows()]  # (row_index, set of columns)
    df[her2enriched_header] = new_column
    if unclear_label is not None:
        unclear_mask = GetUnclearList(df, [hr_header, her2_header], pos, neg)
        df = RenameUnclearMask(df, her2enriched_header, unclear_mask, unclear_label)
    return df

def AddTripleNegativeType(df, hr_header, her2_header, tn_header="TN", pos = "Positive", neg = "Negative", unclear_label = "unknown"):
    new_column = [ pos if row[1][hr_header] == neg and row[1][her2_header] == neg else neg for row in df.iterrows()]  # (row_index, set of columns)
    df[tn_header] = new_column
    if unclear_label is not None:
        unclear_mask = GetUnclearList(df, [hr_header, her2_header], pos, neg)
        df = RenameUnclearMask(df, tn_header, unclear_mask, unclear_label)
    return df

# Function to filter all entries with values different than pos or neg for a specific header
def FilterUnclear(df, header, pos = "Positive", neg = "Negative"):
    return df[((df[header] == pos) | (df[header] == neg))]

# Function to rename all entries with values different than pos or neg to an unclear_label for a specific header
def RenameUnclear(df, header, pos = "Positive", neg = "Negative", unclear_label = "unknown"):
    df.loc[GetUnclear(df, header, pos, neg), header] = unclear_label
    return df

# Function to rename all entries with values different than pos or neg to an unclear_label for a list of header
def RenameUnclearList(df, header_list, pos = "Positive", neg = "Negative", unclear_label = "unknown"):
    for header in header_list:
        df.loc[GetUnclear(df, header, pos, neg), header] = unclear_label
    return df

# Function to rename all masked entries to an unclear_label for a specific header
def RenameUnclearMask(df, header, mask, unclear_label = "unknown"):
    df.loc[mask, header] = unclear_label
    return df

# Function to obtain all entrties with values different than pos or neg for a specific header
def GetUnclear(df, header, pos = "Positive", neg = "Negative"):
    return ((df[header] != pos) & (df[header] != neg))

# Function to obtain all entrties with values different than pos or neg for a list of header
def GetUnclearList(df, header_list, pos = "Positive", neg = "Negative"):
    list_unclear = GetUnclear(df, header_list[0], pos, neg)
    for header in header_list[1:]:
        list_unclear |= GetUnclear(df, header, pos, neg)
    return list_unclear



