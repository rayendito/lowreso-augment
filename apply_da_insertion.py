from loaders.load_model_and_tokenizer import load_model_and_tokenizer

if __name__ == "__main__":
    bert_model_path = "w11wo/javanese-bert-small"
    model, tokenizer = load_model_and_tokenizer(bert_model_path)

    