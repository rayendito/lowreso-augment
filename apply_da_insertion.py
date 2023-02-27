import sys        
sys.path.append('./src')

from loaders.load_data import load_data_nusax_jv, load_data_mono_as_list
from augmentators.augmentator_insertion import augment_parallel_with_insertion

if __name__ == "__main__":
    dataset = load_data_nusax_jv()
    monolingual_corpus_id = load_data_mono_as_list("data/mono/indo_for_selftrain")
    monolingual_corpus_jv = load_data_mono_as_list("data/mono/java_for_backtrans")

    bert_model_path = "w11wo/javanese-bert-small"

    # insertion javanese
    augment_parallel_with_insertion(dataset['train']['javanese'], monolingual_corpus_jv, 'nusax_train', 'indonesian', 'javanese', bert_model_path)

    # insertion indonesian