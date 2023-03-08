import sys        
sys.path.append('./src')

from loaders.load_data import load_all_training_data
from loaders.load_embedding_space import load_embedding
from augmentators.augmentator_embeddingsp import augment_parallel_with_embeddingsp
from simalign import SentenceAligner
from utils.align_sentences import *


import pandas as pd

if __name__ == "__main__":
    src_lang = 'indonesian'
    tgt_lang = 'javanese'

    src_corpus_path = './data/train/id-jv.id.train.untok'
    tgt_corpus_path = './data/train/id-jv.jv.train.untok'

    dataset = load_all_training_data(src_lang, tgt_lang, src_corpus_path, tgt_corpus_path)

    aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")
    
    src_embd_space_path  = 'data/augment_embeddingsp/embedding_models/vectors-id.txt'
    tgt_embd_space_path  = 'data/augment_embeddingsp/embedding_models/vectors-jv.txt'

    src_embeddings, src_id2word, src_word2id = load_embedding(src_embd_space_path)
    tgt_embeddings, tgt_id2word, tgt_word2id = load_embedding(tgt_embd_space_path)

    augmentation_tools = {
        'aligner' : aligner,
        'src_embeddings' : src_embeddings,
        'src_id2word' : src_id2word,
        'src_word2id' : src_word2id,
        'tgt_embeddings' : tgt_embeddings,
        'tgt_id2word' : tgt_id2word,
        'tgt_word2id' : tgt_word2id,
    }

    augment_parallel_with_embeddingsp(dataset, src_lang, tgt_lang, augmentation_tools, runs_per_instance=2, min_similarity=0.5)