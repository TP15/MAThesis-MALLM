from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# === Load model and tokenizer ===
model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # Mistral tokenizer doesn't have pad_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # requires bitsandbytes
    device_map="auto"
)

# === Apply LoRA ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# === Load and preprocess your JSONL dataset ===
dataset_path = "/Users/thomaspathe/Documents/MAThesis-MALLM/Finetunedatasetfolder/combined_output_FT170425.jsonl"
dataset = load_dataset("json", data_files=dataset_path, split="train")

# Format: Instruction-style prompt
def format_prompt(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    return {"text": prompt}

# Apply formatting
dataset = dataset.map(format_prompt)

# Tokenize prompts
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# === Training configuration ===
training_args = TrainingArguments(
    output_dir="./mistral-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    logging_dir="./logs",
    save_total_limit=1,
    save_strategy="epoch",
    fp16=True,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    logging_steps=10,
    report_to="none",
)

# === Trainer setup ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# === Start training ===
trainer.train()

# === Save final model ===
model.save_pretrained("mistral-lora-ft")
tokenizer.save_pretrained("mistral-lora-ft")
