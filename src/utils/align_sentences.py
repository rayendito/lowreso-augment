import numpy as np

def align_sentence_with_aligner(aligner, src_sentenc_split, tgt_sentence_split):
    alignments = aligner.get_word_aligns(src_sentenc_split, tgt_sentence_split)
    return alignments['inter']

def _get_word_vector(word, emb, word2id):
    return emb[word2id[word]]

def _vector_closeness(a, b):
    return (a / np.linalg.norm(a)).dot(b / np.linalg.norm(b))

def align_sentence_with_embedding_space(augmentation_tools, src_sentenc_split, tgt_sentence_split):
    src_embeddings = augmentation_tools['src_embeddings']
    src_word2id = augmentation_tools['src_word2id']
    tgt_embeddings = augmentation_tools['tgt_embeddings']
    tgt_word2id = augmentation_tools['tgt_word2id']

    pairs = []
    for i in range(len(src_sentenc_split)):
        try:
            src_wv = _get_word_vector(src_sentenc_split[i], src_embeddings, src_word2id)
        except KeyError:
            continue
        highest_score = 0
        tgt_best_match_idx = 0
        for j in range(len(tgt_sentence_split)):
            try:
                tgt_wv = _get_word_vector(tgt_sentence_split[j], tgt_embeddings, tgt_word2id)
            except KeyError:
                continue
            current_score = _vector_closeness(src_wv, tgt_wv)
            if (current_score >= highest_score):
                highest_score = current_score
                tgt_best_match_idx = j
        if(highest_score > 0.45):
            pairs.append((i, tgt_best_match_idx))

    return pairs

def evaluate_sentence_alignment(pairs, src_sentenc_split, tgt_sentence_split):
    for pair in pairs:
        print("{} : {}".format(src_sentenc_split[pair[0]], tgt_sentence_split[pair[1]]))