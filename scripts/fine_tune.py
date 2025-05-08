# scripts/fine_tune.py
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

MODEL_NAME = "gpt2"
OUTPUT_DIR = "models/clarified_model"
TRAIN_FILE = "data/clarification_train.txt"


def load_model_and_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    # Add pad token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def prepare_dataset(tokenizer, train_file):
    with open(train_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    dataset = Dataset.from_dict({"text": lines})

    def tokenize_function(example):
        result = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=256
        )
        result["labels"] = result["input_ids"].copy()
        return result

    return dataset.map(tokenize_function, batched=False)


def run_fine_tuning(train_file=TRAIN_FILE):
    model, tokenizer = load_model_and_tokenizer()
    tokenized_dataset = prepare_dataset(tokenizer, train_file)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10,
        save_total_limit=2,
        logging_dir="logs",
        logging_steps=10,
        remove_unused_columns=False  # Required to keep 'text' around
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Fine-tuned model saved to {OUTPUT_DIR}")
