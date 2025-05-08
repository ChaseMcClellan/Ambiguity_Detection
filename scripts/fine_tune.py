# scripts/fine_tune.py

from datasets import Dataset
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)

MODEL_NAME = "gpt2"
OUTPUT_DIR = "models/clarified_model"
TRAIN_FILE = "data/clarification_train.txt"


def load_model_and_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def prepare_dataset(tokenizer, train_file):
    with open(train_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    dataset = Dataset.from_dict({"text": lines})

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )

    return dataset.map(tokenize, batched=False)


def run_fine_tuning(train_file=TRAIN_FILE):
    model, tokenizer = load_model_and_tokenizer()
    dataset = prepare_dataset(tokenizer, train_file)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        learning_rate=5e-5,
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir="logs",
        report_to=[]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Fine-tuned model saved to {OUTPUT_DIR}")
