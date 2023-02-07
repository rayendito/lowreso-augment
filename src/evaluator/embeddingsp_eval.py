from numpy import dot
from numpy.linalg import norm

def _calculate_cosine_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def _calculate_sum_of_similarity(embedding_model, words):
    summed_cosines = 0
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            word_i_vec = embedding_model.get_word_vector(words[i])
            word_j_vec = embedding_model.get_word_vector(words[j])
            summed_cosines += _calculate_cosine_sim(word_i_vec, word_j_vec)
    return summed_cosines

def _calculate_compactness_of_one_word_against_its_group(word, word_group):
    sum_of_similarity = _calculate_sum_of_similarity([x for x in word_group if x != word])
    return sum_of_similarity/(len(word_group)*(len(word_group)-1))

def evaluate_compactness_score(embedding_model, references):
    '''
    reference is a 2D array of words, each element in the outer array is a group of words, with the first element being the outlier
    example of one element from the array
    [
        outlier,
        word,
        word,
        word,
    ]
    '''
    def _evaluate_one_reference(embedding_model, reference):
        compactness_scores = [_calculate_compactness_of_one_word_against_its_group(ref, embedding_model) for ref in reference]
        diff_against_outlier_avg = [compactness_scores[i] - compactness_scores[0] for i in range(1, len(compactness_scores))]
        return sum(diff_against_outlier_avg)/len(diff_against_outlier_avg)

    avg_compactness = []
    for reference in references:
        avg_compactness.append(_evaluate_one_reference(embedding_model, reference))
    
    return sum(avg_compactness)/len(avg_compactness)