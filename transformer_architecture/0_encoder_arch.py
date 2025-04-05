""""
bert-base-uncased model
is a BERT model that is pre-trained on a large corpus of English data in an uncased manner. It is commonly used for various NLP tasks such as text classification, named entity recognition, and question answering.
It is a transformer-based model that uses the encoder architecture of the transformer model. The base version of the model has 12 layers (transformer blocks), 768 hidden dimensions, and 12 attention heads. 
The uncased version means that the model does not differentiate between uppercase and lowercase letters, making it suitable for tasks where case sensitivity is not important.

The model is used for various NLP tasks such as text classification, named entity recognition, and question answering.
The model is trained on a large corpus of English text and is capable of understanding the context and relationships between words in a sentence. It can be fine-tuned for specific tasks by training it on a smaller dataset related to the task at hand.
"""
from transformers import pipeline

#llm = pipeline(model="bert-base-uncased", device=0) # device=0 for GPU, device=-1 for CPU

#print(llm.model) #To determine the model architecture

#print(llm.model.config.is_decoder) #To determine the model architecture
#print(llm.model.config.is_encoder_decoder) #To determine the model architecture

#print(llm.model.config)
#print(llm.model.config.architectures)
#print(llm.model.config.num_hidden_layers)

""""
Text Classification
Sentiment analysis, spam detection, topic classification, etc.
"""
# classifier = pipeline("text-classification", model="bert-base-uncased")
# output = classifier("This movie was fantastic!")
# print(output)  # [{'label': 'LABEL_1', 'score': 0.9998}]


""""
Named Entity Recognition (NER)
Finding names, locations, dates, etc. in text.
"""

# ner = pipeline("ner", model="bert-base-uncased", grouped_entities=True)
# output = ner("Barack Obama was born in Hawaii.")
# print(output) 


""""
Question Answering
Given a passage and a question, return the answer from the text.
"""

# qa = pipeline("question-answering", model="bert-base-uncased")
# output = qa(question="Where was Obama born?", context="Barack Obama was born in Hawaii.")
# print(output)

# qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
# output = qa(
#     question="Where was Obama born?",
#     context="Barack Obama was born in Hawaii."
# )

# print(output)


question = "Who painted the Mona Lisa?"

# Define the appropriate model
qa = pipeline(task="question-answering", model="distilbert-base-uncased-distilled-squad")

text = """"
The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci. Considered an archetypal masterpiece of the Italian Renaissance, it has been described as the most known, visited, talked about, and sung about work of art in the world. The painting's novel qualities include the subject's enigmatic expression, the monumentality of the composition, and the subtle modeling of forms.
"""
output = qa(question=question, context=text)
print(output['answer'])
