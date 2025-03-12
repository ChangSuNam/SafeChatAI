import torch
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load model and tokenizer
model_name = "google/mobilebert-uncased"
tokenizer = MobileBertTokenizer.from_pretrained(model_name)
model = MobileBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Enable MPS if available
if torch.backends.mps.is_available():
    model.to('mps')
    print("Using MPS device")
else:
    print("Using CPU device")

# Load dataset
dataset = load_dataset("csv", data_files="safety_dataset.csv")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", max_length=64, truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)["train"].train_test_split(test_size=0.2)

# Traning arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_mobilebert",
    num_train_epochs=15,
    per_device_train_batch_size=8,    # fits 1000 exampled dataset
    per_device_eval_batch_size=8,     # Matching train batch size
    warmup_steps=500,                 # warmup for 1500 steps (100 steps/epoch × 15)
    weight_decay=0.01,
    learning_rate=5e-5,
    logging_steps=10,
    evaluation_strategy="epoch",      # Eval each epoch, as in last run
    save_strategy="epoch",            # Save each epoch, as in last run
    load_best_model_at_end=True,      # Load best eval loss model, as in last run
    metric_for_best_model="eval_loss",
    greater_is_better=False,          # Lower eval loss is better
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Train and save
trainer.train()
model.save_pretrained("fine_tuned_mobilebert")
tokenizer.save_pretrained("fine_tuned_mobilebert")
print("Model fine-tuning complete!!!")
