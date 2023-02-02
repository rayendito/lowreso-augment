from transformers import    MBartForConditionalGeneration, \
                            MBart50TokenizerFast

def _load_model(model_path):
    if (model_path == 'facebook/mbart-large-50'):
        return MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
    else:
        raise ValueError("{} model path is not recognized/in the scope of this experiment.")

def _load_tokenizer(model_path):
    if (model_path == 'facebook/mbart-large-50'):
        return MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="id_ID", tgt_lang="id_ID")
    else:
        raise ValueError("{} model path is not recognized/in the scope of this experiment.")

def load_model_and_tokenizer(model_path):
    return _load_model(model_path), _load_tokenizer(model_path)