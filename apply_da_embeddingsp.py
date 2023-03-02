import sys        
sys.path.append('./src')

from loaders.load_data import load_data_nusax_jv
from loaders.load_embedding_space import load_embedding
from augmentators.augmentator_embeddingsp import augment_parallel_with_embeddingsp
from simalign import SentenceAligner

if __name__ == "__main__":
    dataset = load_data_nusax_jv()
    src_embd_space_path  = 'data/augment_embeddingsp/embedding_models/vectors-id.txt'
    tgt_embd_space_path  = 'data/augment_embeddingsp/embedding_models/vectors-jv.txt'

    aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")

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

    corpus = dataset['train']
    dataset_name = 'nusax_train'
    src_lang = 'indonesian'
    tgt_lang = 'javanese'

    augment_parallel_with_embeddingsp(corpus, dataset_name, src_lang, tgt_lang, augmentation_tools, runs_per_instance=2)