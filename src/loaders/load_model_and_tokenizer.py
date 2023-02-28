import torch
from transformers import    MBartForConditionalGeneration, \
                            MBart50TokenizerFast, \
                            AutoModel, \
                            AutoTokenizer \

def _load_model(model_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if (model_path == 'facebook/mbart-large-50'):
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
    else:
        model = AutoModel.from_pretrained(model_path)
    
    if(device == 'cuda:0'):
        print("CUDA available, sending model to GPU")
    else:
        print("CUDA not available, model in CPU")
    
    return model.to(device)

def _load_tokenizer(model_path):
    if (model_path == 'facebook/mbart-large-50'):
        return MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="id_ID", tgt_lang="id_ID")
    else:
        return AutoTokenizer.from_pretrained(model_path)

def load_model_and_tokenizer(model_path):
    return _load_model(model_path), _load_tokenizer(model_path)