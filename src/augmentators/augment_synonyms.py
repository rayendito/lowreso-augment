import os
import pandas as pd

def _get_preprocessed_lexicon(src_wordlist, tgt_wordlist):
    # sanity check
    if(len(src_wordlist) != len(tgt_wordlist)):
        raise AssertionError("src and tgt entries unmatched {} and {}".format(len(src_wordlist), len(tgt_wordlist)))
    
    print("Found {} dictionary entries".format(len(src_wordlist)))
    print("Removing same src-tgt entries...")
    
    resulting_dict = {}
    same_count = 0
    for i in range(len(src_wordlist)):
        if(src_wordlist[i] == tgt_wordlist[i]):
            same_count += 1
            continue

        if (src_wordlist[i] in resulting_dict):
            resulting_dict[src_wordlist[i]].add(tgt_wordlist[i])
        else:
            resulting_dict[src_wordlist[i]] = {tgt_wordlist[i]}

    print("Found and removed {} same src-tgt entries.".format(same_count))
    print("Resulting in {} src word entries.".format(len(resulting_dict)))
    return resulting_dict

def load_lexicon(lexicon_file, src_col_name, tgt_col_name):
    lex = pd.read_csv(lexicon_file)
    return _get_preprocessed_lexicon(lex[src_col_name], lex[tgt_col_name])

def _move_word_to_front(word, list_of_words):
    # sanity check lol
    if(word not in list_of_words):
        raise AssertionError("Word \'{}\' not in list of words".format(word))

    list_of_words.remove(word)
    return [word] + list_of_words

def _get_synonym_entries_used(tgt_sent_tokenized, dict_values):
    substitute_sets = {}
    for word_idx in range(len(tgt_sent_tokenized)):
        for synonym_set in dict_values:
            if (tgt_sent_tokenized[word_idx] in synonym_set):
                substitute_sets[word_idx] = _move_word_to_front(tgt_sent_tokenized[word_idx], list(synonym_set))
    return substitute_sets

def _get_all_synonym_permutation(synonyms, tokenized_txt):
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
        

def augment_parallel_with_synonym(corpus, lexicon, src_lang, tgt_lang):

    tgt_path = "./data/augment_synonym/"+"synonym_augmented_{}-{}.csv".format(src_lang, tgt_lang)

    print("Creating new training instances from synonyms...")
    for i in range(2):
        syn_entries = _get_synonym_entries_used(corpus[i][tgt_lang].split(), lexicon.values())
        if(len(syn_entries) == 0):
            continue

        one_entry_augmented = [(corpus[i][src_lang], aug) for aug in _get_all_synonym_permutation(syn_entries, corpus[i][tgt_lang].split())]
        df_augmented = pd.DataFrame(one_entry_augmented, columns =[src_lang, tgt_lang])

        if(os.path.isfile(tgt_path)):
            df_augmented.to_csv(tgt_path, mode='a', index=False, header=False)
        else:
            df_augmented.to_csv(tgt_path, index=False)
    print("Created new training instances in {}".format(tgt_path))