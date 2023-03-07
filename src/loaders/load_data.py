import pandas as pd
from datasets import load_dataset

NUSAX_MAIN = "./data/nusax/"
NUSAX_PATHS = {
        "train": "nusax_jv_train.csv",
        "test": "nusax_jv_test.csv",
        "validation" : "nusax_jv_validation.csv"
    }

KN_MAIN = "./data/korpus_nusantara/"
KN_PATHS = {
    "java" : "kn_jawa.csv",
    "java_ngoko" : "kn_jawa_ngoko.csv",
}

def load_data_nusax_jv(kind = "datasets"):
    if (kind == "datasets"):
        dataset = load_dataset(NUSAX_MAIN, data_files=NUSAX_PATHS)
        return dataset
    elif (kind == "dataframe"):
        frames = [pd.read_csv(NUSAX_MAIN+NUSAX_PATHS["test"]), pd.read_csv(NUSAX_MAIN+NUSAX_PATHS["train"]), pd.read_csv(NUSAX_MAIN+NUSAX_PATHS["validation"])]
        return pd.concat(frames)
    raise AssertionError("{} format not recognized".format(kind))

def load_data_kn_jv_as_df():
    names = ['indonesian', 'javanese']
    frames = [pd.read_csv(KN_MAIN+KN_PATHS["java"], names=names), pd.read_csv(KN_MAIN+KN_PATHS["java_ngoko"], names=names)]
    return pd.concat(frames)

def load_all_training_data():
    nusax = load_data_nusax_jv(kind="dataframe")
    kn = load_data_kn_jv_as_df()
    training_corpuses = [nusax, kn]
    return pd.concat(training_corpuses)

def load_data_mono_as_list(path):
    myfile = open(path, "r", encoding="utf8")
    data = myfile.read()
    myfile.close()
    return data.split("\n")