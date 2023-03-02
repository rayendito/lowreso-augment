def align_sentence_with_aligner(aligner, src_sentenc_split, tgt_sentence_split):
    alignments = aligner.get_word_aligns(src_sentenc_split, tgt_sentence_split)
    return alignments['inter']

def align_sentence_with_embedding_space(augmentation_tools, src_sent, tgt_sent):
    pass