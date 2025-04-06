from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Check if GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Load the model and move it to the GPU
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)


# Load the IMDb dataset
dataset = load_dataset("imdb")
    
num_shards = 4

for i in range(num_shards):
    print(f"\n Processing Shard {i + 1}/{num_shards}...")
    
    train_dataset = dataset["train"].shard(num_shards=num_shards, index=i)
    test_dataset = dataset["test"].shard(num_shards=num_shards, index=i)
    
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    # Freeze all layers except the classification head
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Define a compute metrics function
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1).cpu()
        precision, recall, f1, _ = precision_recall_fscore_support(torch.tensor(labels).cpu(), predictions, average="binary")
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
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