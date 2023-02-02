from transformers import Seq2SeqTrainingArguments


def load_training_args():
    training_args = Seq2SeqTrainingArguments(
        output_dir="./../dumps",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True, # half-precision
    )

    return training_args