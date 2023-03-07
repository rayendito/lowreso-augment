import pandas as pd
from datasets import load_dataset

def load_data_mono_as_list(path):
    myfile = open(path, "r", encoding="utf8")
    data = myfile.read()
    myfile.close()
    return data.split("\n")

def load_all_training_data(src_lang_name, tgt_lang_name, src_lang_file, tgt_lang_file):
    src_sents = load_data_mono_as_list(src_lang_file)
    tgt_sents = load_data_mono_as_list(tgt_lang_file)
    df = pd.DataFrame(list(zip(src_sents, tgt_sents)), columns =[src_lang_name, tgt_lang_name])
    return df