from datasets import load_dataset

def load_data_nusax_jv():
    data_files = {
        "train": "nusax_jv_train.csv",
        "test": "nusax_jv_test.csv",
        "validation" : "nusax_jv_validation.csv"
    }

    dataset = load_dataset("./data/nusax", data_files=data_files)
    return dataset