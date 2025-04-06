from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# Load the pre-trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_single_text(text, padding=True, truncation=True, max_length=64):
    return tokenizer(
        text, 
        return_tensors="pt", 
        padding=padding, 
        truncation=truncation, 
        max_length=max_length
    )

# Example usage
example_text = "This is an example review text."
tokenized_text = tokenize_single_text(example_text)
print(tokenized_text)
#print(tokenizer.decode([101, 2023, 2003, 2019, 2742, 3319, 3793, 1012, 102]))