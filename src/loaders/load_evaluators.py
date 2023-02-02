import evaluate
import numpy as np

def _postprocess_text(preds, labels):
  preds = [pred.strip() for pred in preds]
  labels = [[label.strip()] for label in labels]
  return preds, labels

def get_compute_metrics_function(tokenizer):
    sacrebleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # if label is -100 turn into padding token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # decoding
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = _postprocess_text(decoded_preds, decoded_labels)

        # get BLEU score
        result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        # get length stats of predicitons
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)

        # rounding everything
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    return compute_metrics