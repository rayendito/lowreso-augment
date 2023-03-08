import os
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
from loaders.load_embedding_space import load_embedding
from utils.align_sentences import align_sentence_with_aligner, align_sentence_with_embedding_space

def _get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=5):
    word2id = {v: k for k, v in src_id2word.items()}
    word_emb = src_emb[word2id[word]]
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    return [(scores[idx], tgt_id2word[idx]) for i, idx in enumerate(k_best)]

def _clean_token(tok):
    return re.sub(r'\W+', '', tok).lower().strip()

def _get_word_vector(word, emb, word2id):
    return emb[word2id[word]]

def _vector_closeness(a, b):
    return (a / np.linalg.norm(a)).dot(b / np.linalg.norm(b))

def _augment_one_instance_with_embeddingsp(src_sent, tgt_sent, augmentation_tools, min_similarity):
    # unpacking tools
    aligner = augmentation_tools['aligner']
    src_embeddings = augmentation_tools['src_embeddings']
    src_id2word = augmentation_tools['src_id2word']
    src_word2id = augmentation_tools['src_word2id']
    tgt_embeddings = augmentation_tools['tgt_embeddings']
    tgt_id2word = augmentation_tools['tgt_id2word']
    tgt_word2id = augmentation_tools['tgt_word2id']

    src_sent_split = src_sent.split()
    tgt_sent_split = tgt_sent.split()

    # get alignment
    if(aligner == "None"):
        alignment_indices = align_sentence_with_embedding_space(aligner, src_sent_split, tgt_sent_split)
    else:
        alignment_indices = align_sentence_with_aligner(aligner, src_sent_split, tgt_sent_split)

    def __pair_scores_sorted(src_words, tgt_words):
        if (len(src_words) != len(tgt_words)):
            return AssertionError("src words and tgt words length not the same")
        
        pairs = []
        for i_word in src_words:
            for j_word in tgt_words:
                src_wv = _get_word_vector(i_word, src_embeddings, src_word2id)
                tgt_wv = _get_word_vector(j_word, tgt_embeddings, tgt_word2id)
                sim_score = _vector_closeness(src_wv, tgt_wv)
                if(sim_score >= min_similarity):
                    pairs.append([sim_score, i_word, j_word])

        return sorted(pairs, key=lambda x: x[0], reverse=True)

    for pair in alignment_indices:
        try:
            neighbors_src = _get_nn(_clean_token(src_sent_split[pair[0]]), src_embeddings, src_id2word, src_embeddings, src_id2word, K=10)
            neighbors_tgt = _get_nn(_clean_token(tgt_sent_split[pair[1]]), tgt_embeddings, tgt_id2word, tgt_embeddings, tgt_id2word, K=10)
        except KeyError: # if the token is not found in src or tgt
            continue

        neighbors_src_tokens = [neighbor[1] for neighbor in neighbors_src]
        neighbors_tgt_tokens = [neighbor[1] for neighbor in neighbors_tgt]

        pairs = __pair_scores_sorted(neighbors_src_tokens, neighbors_tgt_tokens)
        if(len(pairs) != 0):
            chosen_pair = pairs[np.random.randint(len(pairs))]
            src_sent_split[pair[0]] = chosen_pair[1]
            tgt_sent_split[pair[1]] = chosen_pair[2]

    return " ".join(src_sent_split), " ".join(tgt_sent_split)

def augment_parallel_with_embeddingsp(corpus, src_lang, tgt_lang, augmentation_tools = 'None', runs_per_instance=2, min_similarity=0.4):
    src_file_path = "./data/augment_embeddingsp/"+"{}-{}.{}.untok".format(src_lang, tgt_lang, src_lang)
    tgt_file_path = "./data/augment_embeddingsp/"+"{}-{}.{}.untok".format(src_lang, tgt_lang, tgt_lang)
    print("Creating new training instances from embedding space...")
    
    src_f = open(src_file_path, "a+", encoding="utf8")
    tgt_f = open(tgt_file_path, "a+", encoding="utf8")
    for index, row in tqdm(corpus.iterrows()):
        for j in range(runs_per_instance):
            augmented_src, augmented_tgt = _augment_one_instance_with_embeddingsp(row[src_lang], row[tgt_lang], augmentation_tools, min_similarity)
            src_f.write(augmented_src+"\n")
            tgt_f.write(augmented_tgt+"\n")
    src_f.close()
    tgt_f.close()

    print("Created new training instances in {} and {}".format(src_file_path, tgt_file_path))