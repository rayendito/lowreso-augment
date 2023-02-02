# import sys, os
# sys.path.append(os.path.abspath(os.path.join('..')))

import pandas as pd
from loaders.load_data import load_data_nusax_jv

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

def augment_parallel_with_synonym(corpus, processed_lexicon):
    for i in range(len(corpus)):
        print(corpus[i])
        break