
from transformers import pipeline

llm = pipeline(model="gpt2")

#print(llm.model) #To determine the model architecture
#print(llm.model.config) #text generation #To determine the model architecture
#print(llm.model.config.is_decoder) #returns False while gpt2 is a decoder-only model


#"gpt-3.5-turbo"

"""
The gpt2 model is a decoder-only transformer model, which means it is designed to generate text based on a given input. 
It does not have an encoder component like the BERT model, which is designed for understanding and processing text.
The gpt2 model is trained to predict the next word in a sequence, given the previous words. This is done using a causal language 
modeling objective, where the model learns to generate text by predicting the next word in a sequence based on the previous words. 
The gpt2 model is also pre-trained on a large corpus of text data, which allows it to generate coherent and contextually relevant text.

The gpt2 model is a transformer-based model, which means it uses self-attention mechanisms to process and generate text.
The self-attention mechanism allows the model to weigh the importance of different words in a sequence when generating text,

which helps it to generate more coherent and contextually relevant text.
The gpt2 model is also designed to be fine-tuned on specific tasks, such as text generation, summarization, and question answering.

The gpt2 model is a powerful tool for generating text, and it has been used in a variety of applications, including chatbots, content generation, and creative writing.
The gpt2 model is a state-of-the-art language model that has been widely adopted in the field of natural language processing.

the tokenizer is used to convert text into a format that the model can understand, and the model is used to generate text based on the input. The
tockenizer is responsible for breaking down the input text into smaller units, called tokens, which are then fed into the model for processing.
The model uses these tokens to generate text by predicting the next token in the sequence based on the previous tokens. 
The generated text is then converted back into human-readable format using the tokenizer.
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to the end-of-sequence token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

prompt = """"
Task: Given a text, generate a summary of the text.
Input: which helps it to generate more coherent and contextually relevant text.
The gpt2 model is also designed to be fine-tuned on specific tasks, such as text generation, summarization, and question answering.

The gpt2 model is a powerful tool for generating text, and it has been used in a variety of applications, including chatbots, content generation, and creative writing.
The gpt2 model is a state-of-the-art language model that has been widely adopted in the field of natural language processing.

the tokenizer is used to convert text into a format that the model can understand, and the model is used to generate text based on the input. The
tockenizer is responsible for breaking down the input text into smaller units, called tokens, which are then fed into the model for processing.
The model uses these tokens to generate text by predicting the next token in the sequence based on the previous tokens. 
The generated text is then converted back into human-readable format using the tokenizer.
"""

inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=500, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, num_return_sequences=1)
print(tokenizer.decode(outputs[0]))