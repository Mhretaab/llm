""""
bert-base-uncased model
is a BERT model that is pre-trained on a large corpus of English data in an uncased manner. It is commonly used for various NLP tasks such as text classification, named entity recognition, and question answering.
It is a transformer-based model that uses the encoder architecture of the transformer model. The base version of the model has 12 layers (transformer blocks), 768 hidden dimensions, and 12 attention heads. 
The uncased version means that the model does not differentiate between uppercase and lowercase letters, making it suitable for tasks where case sensitivity is not important.

The model is used for various NLP tasks such as text classification, named entity recognition, and question answering.
The model is trained on a large corpus of English text and is capable of understanding the context and relationships between words in a sentence. It can be fine-tuned for specific tasks by training it on a smaller dataset related to the task at hand.
"""
from transformers import pipeline

llm = pipeline(model="bert-base-uncased")

#print(llm.model) 
#print(llm.model.config)
print(llm.model.config.is_decoder)

