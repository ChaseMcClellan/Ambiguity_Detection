from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import logging
import torch

logging.set_verbosity_error()

MODEL_NAME = "gpt2"
TRAIN_FILE = "../data/clarification_train.txt"
OUTPUT_DIR = "./fine_tuned_model"

#
def load_custom_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n\n")
    return [{"text": line.strip()} for line in lines if line.strip()]

raw_data = load_custom_dataset(TRAIN_FILE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  #gpt2

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

#dataset
from datasets import Dataset
dataset = Dataset.from_list(raw_data)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

#load
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

#config
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    warmup_steps=5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    save_total_limit=1,
    save_strategy="no",
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train
trainer.train()

# Save final model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
