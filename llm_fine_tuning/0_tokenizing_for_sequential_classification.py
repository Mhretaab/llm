from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# Load the IMDB dataset
train_data = load_dataset("imdb", split="train")

# print(f"Number of training samples: {len(train_data)}")
# print(f"Type of data: {type(train_data)}")
# print(f"First sample: {train_data[0]}")
#print(f"First sample text: {train_data[0]['text']}")

train_data = train_data.shard(num_shards=4, index=0)
test_data = load_dataset("imdb", split="test")
test_data = test_data.shard(num_shards=4, index=0)

# Load the pre-trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the data
tokenized_training_data = tokenizer(
	train_data["text"], return_tensors="pt", padding=True, truncation=True, max_length=64
)
tokenized_test_data = tokenizer(
	test_data["text"], return_tensors="pt", padding=True, truncation=True, max_length=64
)

# Print the shapes of the tokenized data
#print(tokenized_training_data["input_ids"].shape)
print(tokenized_test_data)

# print(tokenized_training_data["input_ids"])
# print(tokenized_training_data["attention_mask"])


# # Print the first tokenized sample
# print(tokenized_training_data["input_ids"][0])