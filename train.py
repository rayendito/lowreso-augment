import sys        
sys.path.append('./src')

import argparse
from transformers import Seq2SeqTrainer
from loaders.load_data import load_data_nusax
from loaders.load_model_and_tokenizer import load_model_and_tokenizer
from loaders.load_data_collator import load_data_collator
from loaders.load_evaluators import get_compute_metrics_function
from utils.tokenize import get_tokenizer_function, tokenize
from training_args import load_training_args

parser = argparse.ArgumentParser(
                    prog = 'Indonesian-Javanese Machine Translation fine-tune-r',
                    description = 'It finetunes various pretrained models',
                    epilog = 'Hoy acaba algo, pero es el primer dia de tu siguiente vida!')

parser.add_argument('model_name')

args = parser.parse_args()

MODEL_PATHS = {
    "mbart50" : "facebook/mbart-large-50"
}

print("Loading data...")
dataset = load_data_nusax()

print("Loading model and tokenizer...")
model, tokenizer = load_model_and_tokenizer(MODEL_PATHS[args.model_name])
training_args = load_training_args()

tokenizing_function = get_tokenizer_function(tokenizer)
tokenized_dataset = tokenize(dataset, tokenizing_function)

data_collator = load_data_collator(tokenizer, model)

compute_metrics = get_compute_metrics_function(tokenizer)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# dor training
print("Finetuning {}...".format(args.model_name))
trainer.train()