from transformers import    MBartForConditionalGeneration, \
                            MBart50TokenizerFast \
                            AutoModel \
                            AutoTokenizer \

def _load_model(model_path):
    if (model_path == 'facebook/mbart-large-50'):
        return MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
    else:
        return AutoModel.from_pretrained(model_path)

def _load_tokenizer(model_path):
    if (model_path == 'facebook/mbart-large-50'):
        return MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="id_ID", tgt_lang="id_ID")
    else:
        raise AutoTokenizer.from_pretrained(model_path)

def load_model_and_tokenizer(model_path):
    return _load_model(model_path), _load_tokenizer(model_path)