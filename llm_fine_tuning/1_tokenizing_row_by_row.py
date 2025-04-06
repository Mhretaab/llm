from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# Load the IMDB dataset
train_data = load_dataset("imdb", split="train")
train_data = train_data.shard(num_shards=4, index=0)
# print(f"Number of training samples: {len(train_data)}")
# print(f"Type of data: {type(train_data)}")
# print(f"First sample: {train_data[0]}")
# print(f"First sample text: {train_data[0]['text']}")
test_data = load_dataset("imdb", split="test")
test_data = test_data.shard(num_shards=4, index=0)

# Load the pre-trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(text_data):
	return tokenizer(
		text_data["text"], 
		return_tensors="pt", 
		padding=True, 
		truncation=True, 
		max_length=64
	)

# Tokenize in batches
tokenized_in_batches = train_data.map(tokenize_function, batched=True)
print(tokenized_in_batches)
# print(tokenized_in_batches[0])
# print(tokenized_in_batches[0]["input_ids"])

# Tokenize row by row
tokenized_by_row = train_data.map(tokenize_function, batched=False)
print(tokenized_by_row)
# print(tokenized_in_batches[0])
# print(tokenized_by_row[0])
