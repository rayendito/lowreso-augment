from transformers import DataCollatorForSeq2Seq

def load_data_collator(tokenizer, model):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)