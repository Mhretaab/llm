from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch  # Required for inference

# Example usage to avoid unused import errors
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Ensure the model is moved to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the IMDB dataset
train_data = load_dataset("imdb", split="train")
#train_data = train_data.shard(num_shards=4, index=0)

test_data = load_dataset("imdb", split="test")
#test_data = test_data.shard(num_shards=4, index=0)


def tokenize_function(text_data):
    return tokenizer(
        text_data["text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

# Tokenize in batches
tokenized_train_data = train_data.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_test_data = test_data.map(tokenize_function, batched=True, remove_columns=["text"])

# Define the training arguments for fine-tuning the model
training_args = TrainingArguments(
    output_dir="./finetuned",  # Directory to save the fine-tuned model and checkpoints
    evaluation_strategy="epoch",  # Evaluate the model at the end of each epoch
    logging_dir="./logs",
    logging_steps=500,  # log every 500 steps
    num_train_epochs=3,  # Number of training epochs
    learning_rate=2e-5,  # Learning rate for the optimizer
    per_device_train_batch_size=8,  # Batch size for training on each device
    per_device_eval_batch_size=8,  # Batch size for evaluation on each device
    weight_decay=0.01,  # Weight decay for regularization to prevent overfitting
    # Removed the device argument as it is not supported by TrainingArguments
)

# Initialize the Trainer with the model, training arguments, datasets, and tokenizer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_test_data,
    tokenizer=tokenizer,
    compute_metrics=lambda p: {
        "accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()
    }
)

trainer.train()
trainer.evaluate()

new_data = ["This movie was disappointing!", "This is the best movie ever!"]
new_input = tokenizer(new_data, return_tensors="pt", padding=True, truncation=True, max_length=64).to(model.device)

# Perform inference on the new input data without computing gradients (saves memory and speeds up computation)
with torch.no_grad():
    outputs = model(**new_input)


predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()

label_map = {0: "NEGATIVE", 1: "POSITIVE"}
for i, predicted_label in enumerate(predicted_labels):
    sentiment = label_map[predicted_label]
    print(f"\nInput Text {i + 1}: {new_data[i]}")
    print(f"Predicted Label: {sentiment}")
    
# Save the fine-tuned model and tokenizer to the specified directory
# This allows the model and tokenizer to be reloaded later for inference or further fine-tuning
model.save_pretrained("my_finetuned_files")
tokenizer.save_pretrained("my_finetuned_files")