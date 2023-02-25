from simalign import SentenceAligner

myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")

def align_sentence_unique(src_sentence, tgt_sentence):
    src_sentence = src_sentence.split()
    tgt_sentence = tgt_sentence.split()
    alignments = myaligner.get_word_aligns(src_sentence, tgt_sentence)
    return alignments['inter']