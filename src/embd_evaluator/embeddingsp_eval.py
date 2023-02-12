from numpy import dot
from numpy.linalg import norm

def _calculate_cosine_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def _calculate_mean_of_similarity(embedding_model, words):
    summed_cosines = []
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            word_i_vec = embedding_model.get_word_vector(words[i])
            word_j_vec = embedding_model.get_word_vector(words[j])
            summed_cosines.append(_calculate_cosine_sim(word_i_vec, word_j_vec))
    return sum(summed_cosines)/len(summed_cosines)

def _calculate_compactness_of_one_word_against_its_group(word, word_group, embedding_model):
    return _calculate_mean_of_similarity(embedding_model, [x for x in word_group if x != word])

def evaluate_compactness_score(embedding_model, references):
    '''
    reference is a 2D array of words, each element in the outer array is a group of words, with the first element being the outlier
    EXAMPLE OF ONE ELEMENT FROM THE ARRAY:
    [
        outlier,
        word,
        word,
        word,
        word,
    ]

    the bigger the score, the better because it shows how different the outlier and the rest of the group
    '''
    def _evaluate_one_reference(embedding_model, reference):
        compactness_scores = [_calculate_compactness_of_one_word_against_its_group(wrd, reference, embedding_model) for wrd in reference]
        diff_against_outlier_avg = [compactness_scores[i] - compactness_scores[0] for i in range(1, len(compactness_scores))]
        return sum(diff_against_outlier_avg)/len(diff_against_outlier_avg)

    avg_compactness = []
    for reference in references:
        avg_compactness.append(_evaluate_one_reference(embedding_model, reference))
    
    return sum(avg_compactness)/len(avg_compactness)