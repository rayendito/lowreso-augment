
import sys        
sys.path.append('./src')

import pandas as pd
import numpy as np
from loaders.load_data import load_data_mono_as_list, load_all_training_data

# def train_validate_test_split(df, train_percent=.8, validate_percent=.1, seed=None):
#     np.random.seed(seed)
#     perm = np.random.permutation(df.index)
#     m = len(df.index)
#     train_end = int(train_percent * m)
#     validate_end = int(validate_percent * m) + train_end
#     train = df.iloc[perm[:train_end]]
#     validate = df.iloc[perm[train_end:validate_end]]
#     test = df.iloc[perm[validate_end:]]
#     return train, validate, test

def list_to_file(lis, target):
    fil = open(target,'w+', encoding="utf8")
    for item in lis:
        fil.write(item+"\n")
    fil.close()

# df = load_all_training_data()
# df = df.replace('\n','', regex=True)

df = pd.read_csv("jv_train_compiled.csv")
print(len(df['indonesian']) == len(df['javanese']))
test = 7099
# print(df['indonesian'][test])
# print(df['javanese'][test])
list_to_file(df['indonesian'], './data/train/id-jv.id.train.untok')
list_to_file(df['javanese'], './data/train/id-jv.jv.train.untok')

# paths = {
#     "ind_dev" : "./data/flores/flores101_dataset/dev/ind.dev",
#     "ind_devtest" : "./data/flores/flores101_dataset/devtest/ind.devtest",
#     "jav_dev" : "./data/flores/flores101_dataset/dev/jav.dev",
#     "jav_devtest" : "./data/flores/flores101_dataset/devtest/jav.devtest",
# }

# indo_sentences = load_data_mono_as_list(paths["ind_dev"]) + load_data_mono_as_list(paths["ind_devtest"])
# java_sentences = load_data_mono_as_list(paths["jav_dev"]) + load_data_mono_as_list(paths["jav_devtest"])

# if(len(indo_sentences) != len(java_sentences)):
#     raise AssertionError("panjangnya ngg sama coy")

# parallel = [(indo_sentences[i], java_sentences[i]) for i in range(len(indo_sentences))]
# df = pd.DataFrame(parallel, columns =['ind', 'jav'])


# train, validate, test = train_validate_test_split(df, seed=69)

# TARGET_DIR = "data/flores/ind_jav_split/"

# splits = {
#     'train' : train,
#     'test' : test,
#     'validation' : validate,
# }

# for split in splits:
#     for lang in ['ind', 'jav']:
#         list_to_file(list(splits[split][lang]), "{}{}.{}".format(TARGET_DIR, lang, split))

