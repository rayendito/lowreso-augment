import os
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
from loaders.load_embedding_space import load_embedding
from utils.align_sentences import align_sentence_unique

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

def _augment_one_instance_with_embeddingsp(src_sent, tgt_sent):
    src_embd_space_path  = 'data/augment_embeddingsp/embedding_models/vectors-id.txt'
    tgt_embd_space_path  = 'data/augment_embeddingsp/embedding_models/vectors-jv.txt'
    
    src_embeddings, src_id2word, src_word2id = load_embedding(src_embd_space_path)
    tgt_embeddings, tgt_id2word, tgt_word2id = load_embedding(tgt_embd_space_path)
    
    alignment_indices = align_sentence_unique(src_sent, tgt_sent)

    src_sent_tok = src_sent.split()
    tgt_sent_tok = tgt_sent.split()

    def _pair_scores_sorted(src_words, tgt_words):
        if (len(src_words) != len(tgt_words)):
            return AssertionError("src words and tgt words length not the same")
        
        pairs = []
        for i_word in src_words:
            for j_word in tgt_words:
                src_wv = _get_word_vector(i_word, src_embeddings, src_word2id)
                tgt_wv = _get_word_vector(j_word, tgt_embeddings, tgt_word2id)
                sim_score = _vector_closeness(src_wv, tgt_wv)
                if(sim_score >= 0.4):
                    pairs.append([sim_score, i_word, j_word])

        return sorted(pairs, key=lambda x: x[0], reverse=True)

    for pair in alignment_indices:
        try:
            neighbors_src = _get_nn(_clean_token(src_sent_tok[pair[0]]), src_embeddings, src_id2word, src_embeddings, src_id2word, K=5)
            neighbors_tgt = _get_nn(_clean_token(tgt_sent_tok[pair[1]]), tgt_embeddings, tgt_id2word, tgt_embeddings, tgt_id2word, K=5)
        except KeyError:
            continue

        neighbors_src_tokens = [neighbor[1] for neighbor in neighbors_src]
        neighbors_tgt_tokens = [neighbor[1] for neighbor in neighbors_tgt]

        pairs = _pair_scores_sorted(neighbors_src_tokens, neighbors_tgt_tokens)
        if(len(pairs) != 0):
            chosen_pair = pairs[np.random.randint(len(pairs))]
            src_sent_tok[pair[0]] = chosen_pair[1]
            tgt_sent_tok[pair[1]] = chosen_pair[2]

    return " ".join(src_sent_tok), " ".join(tgt_sent_tok)

def augment_parallel_with_embeddingsp(corpus, dataset_name, src_lang, tgt_lang, runs_per_instance=2):
    tgt_path = "./data/augment_embeddingsp/"+"embeddingsp_augmented_{}_{}-{}.csv".format(dataset_name, src_lang, tgt_lang)
    print("Creating new training instances from embedding space...")
    for i in tqdm(range(len(corpus))):
        new_instances = []
        for j in range(runs_per_instance):
            augmented_src, augmented_tgt = _augment_one_instance_with_embeddingsp(corpus[i][src_lang], corpus[i][tgt_lang])
            new_instances.append((augmented_src, augmented_tgt))

        df_augmented = pd.DataFrame(new_instances, columns =[src_lang, tgt_lang])
        if(os.path.isfile(tgt_path)):
            df_augmented.to_csv(tgt_path, mode='a', index=False, header=False)
        else:
            df_augmented.to_csv(tgt_path, index=False)

    print("Created new training instances in {}".format(tgt_path))