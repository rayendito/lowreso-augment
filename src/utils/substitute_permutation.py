def get_all_substitution_permutation(synonyms, tokenized_txt):
    dict_counter = {idx : {key : 0} for idx, key in enumerate(synonyms.keys())}
    def _get_counter_idx_key(idx):
        return list(dict_counter[idx])[0]

    def _all_not_at_max():
        for i in dict_counter:
            if(dict_counter[i][_get_counter_idx_key(i)] < len(synonyms[_get_counter_idx_key(i)])-1):
                return True
        return False

    def _up_counter():
        dict_counter[0][_get_counter_idx_key(0)] += 1
        for i in range(len(dict_counter)-1):
            if(dict_counter[i][_get_counter_idx_key(i)] == len(synonyms[_get_counter_idx_key(i)])):
                dict_counter[i][_get_counter_idx_key(i)] = 0
                dict_counter[i+1][_get_counter_idx_key(i+1)] += 1

    def _replace_words_based_on_counter():
        for i in dict_counter:
            replace_word_with = dict_counter[i][_get_counter_idx_key(i)]
            tokenized_txt[_get_counter_idx_key(i)] = synonyms[_get_counter_idx_key(i)][replace_word_with]

    new_sentences = []
    while(_all_not_at_max()):
        # up counter
        _up_counter()
        
        # replace
        _replace_words_based_on_counter()

        # append to new_sents
        new_sentences.append(" ".join(tokenized_txt))
    
    return new_sentences