import sys        
sys.path.append('./src')

from loaders.load_data import load_data_nusax_jv
from augmentators.augmentator_embeddingsp import augment_parallel_with_embeddingsp

if __name__ == "__main__":
    dataset = load_data_nusax_jv()
    augment_parallel_with_embeddingsp(dataset['train'], 'nusax_train', 'indonesian', 'javanese', runs_per_instance=2)