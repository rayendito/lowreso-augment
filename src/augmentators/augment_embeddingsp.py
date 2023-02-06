import os
import fasttext
import re

def train_fasttext_model(corpus_path, method = "cbow", minn = 2, maxn=4, epoch = 20, lr = 0.05):
    print("=============TRAINING A WORD EMBEDDING MODEL=============")
    target_path = './data/augment_embeddingsp/embedding_models/jv.{}.{}.300.bin'.format(method, epoch)
    if(os.path.isfile(target_path)):
        print("fasttext model {} already exist! returning...\n".format(target_path))
        return target_path

    print("Training model in progress...")
    model = fasttext.train_unsupervised(corpus_path, method, minn=minn, maxn=maxn, epoch=epoch, lr=lr)
    model.save_model(target_path)
    print("Model saved to {}...".format(target_path))
    print("Naming convention: jv.<method>.<#epoch>.<dimensions>.bin\n")
    return target_path
    
def _load_fasttext_model(path):
    return fasttext.load_model(path)

def _get_substitute_group(sentence,model):
    word_groups = {}
    sentence_cleaned_tokenized =  re.sub(r'[^A-Za-z0-9 -]+', '', sentence).lower().split()
    for i in range(len(sentence_cleaned_tokenized)):
        nearest_neighbors = model.get_nearest_neighbors(sentence_cleaned_tokenized[i])
        word_groups[i] = [neighbor[1] for neighbor in nearest_neighbors if neighbor[0] >= 0.75]
        print(sentence_cleaned_tokenized[i], word_groups[i])
    return word_groups

def substitute_with_embedding(corpus, model_path):
    print("=============SUBSTITUTING WITH A WORD EMBEDDING MODEL=============")
    if(not os.path.isfile(model_path)):
        raise AssertionError("model {} does not exist! returning...".format(model_path))
    
    print("Loading model {}...".format(model_path))
    model = _load_fasttext_model(model_path)

    print("Creating new sentences from embedding space...")
    for sentence in corpus:
        substitute_group = _get_substitute_group(sentence,model)
        break

