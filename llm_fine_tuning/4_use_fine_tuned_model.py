from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("my_finetuned_files")
tokenizer = AutoTokenizer.from_pretrained("my_finetuned_files")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

new_data = ["This movie was disappointing!", "This is the best movie ever!", "I didn't like the plot.", "The acting was superb!"]

# Tokenize the input data
inputs = tokenizer(new_data, padding=True, truncation=True, return_tensors="pt").to(device)

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()

label_map = {0: "NEGATIVE", 1: "POSITIVE"}
for i, predicted_label in enumerate(predicted_labels):
    sentiment = label_map[predicted_label]
    print(f"\nInput Text {i + 1}: {new_data[i]}")
    print(f"Predicted Label: {sentiment}")
    
    
# Example for a single text input
single_text = "The storyline was captivating and engaging."

# Tokenize the single input text
single_input = tokenizer(single_text, padding=True, truncation=True, return_tensors="pt").to(device)

# Perform inference
with torch.no_grad():
    single_output = model(**single_input)

predicted_label = torch.argmax(single_output.logits, dim=1).item()

sentiment = label_map[predicted_label]
print(f"\nInput Text: {single_text}")
print(f"Predicted Label: {sentiment}")