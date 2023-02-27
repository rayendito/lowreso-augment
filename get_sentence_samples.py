import random

def get_sentence_samples(filename, target, n_samples=1000, min_words = 10):
    lines = [line for line in open(filename, errors="ignore") if len(line.split()) >= min_words]
    if(n_samples > len(lines)):
        raise AssertionError("n_samples requested is more than available sentences")
    
    sentences = random.sample(lines, n_samples)
    newfile = open(target,'w')
    for sentence in sentences:
        newfile.write(sentence)
    newfile.close()


if __name__ == "__main__":
    # get_sentence_samples('data/mono/indo_wiki_sample_10k', 'indo_selftrain')
    get_sentence_samples('data/mono/java_wiki_sample_10k', 'java_backtrans')