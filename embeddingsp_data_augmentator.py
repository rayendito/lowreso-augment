import sys        
sys.path.append('./src')

import re
from loaders.load_data import load_data_nusax_jv
from augmentators.augment_embeddingsp import substitute_with_embedding, train_fasttext_model_return_path

if __name__ == "__main__":
    dataset = load_data_nusax_jv()
    model_paths = {
        "cbow_2" : "./data/augment_embeddingsp/embedding_models/cbow/jv.cbow.2.300.bin",
        "cbow_68" : "./data/augment_embeddingsp/embedding_models/cbow/jv.cbow.68.300.bin",
        "skipgram_2" : "./data/augment_embeddingsp/embedding_models/skipgram/jv.skipgram.2.300.bin",
        "skipgram_68" : "./data/augment_embeddingsp/embedding_models/skipgram/jv.skipgram.68.300.bin",
    }

    substitute_with_embedding(dataset['train']['javanese'][51:52], model_paths["cbow_2"], "augment_result_cbow_2.txt")
    