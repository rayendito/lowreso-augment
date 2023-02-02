def get_tokenizer_function(tokenizer, src_lang="indonesian", tgt_lang="javanese"):
    if(tokenizer.name_or_path == "facebook/mbart-large-50"):
        def mbart50_tokenizer(examples):
            inputs = [sent for sent in examples[src_lang]]
            targets = [sent for sent in examples[tgt_lang]]
            return tokenizer.prepare_seq2seq_batch(src_texts=inputs, src_lang="id_ID", tgt_lang="id_ID", tgt_texts=targets, padding='max_length', truncation=True)        
        return mbart50_tokenizer
    else:
        raise ValueError("faulty at tokenizer or examples, sorry this is not a very helpful error message <3 :v")

def tokenize(dataset, tokenize_function):
    return dataset.map(tokenize_function, batched=True)