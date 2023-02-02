import sys        
sys.path.append('./src')

from loaders.load_data import load_data_nusax_jv
from augmentators.augment_synonyms import augment_parallel_with_synonym, load_lexicon

if __name__ == "__main__":
    dataset = load_data_nusax_jv()
    lx = load_lexicon('./data/javanese_lexicon.csv', 'indonesian', 'javanese')
    augment_parallel_with_synonym(dataset['validation'], lx)