import sys        
sys.path.append('./src')

from loaders.load_data import load_data_nusax_jv
from augmentators.augment_synonyms import augment_parallel_with_synonym, load_lexicon
import numpy as np

if __name__ == "__main__":
    dataset = load_data_nusax_jv()
    lx = load_lexicon('./data/augment_synonym/javanese_lexicon.csv', 'indonesian', 'javanese')
    augment_parallel_with_synonym(dataset['train'], lx, 'nusax_train', 'indonesian', 'javanese')