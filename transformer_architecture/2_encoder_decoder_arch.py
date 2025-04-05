from transformers import pipeline

llm = pipeline(model="Helsinki-NLP/opus-mt-es-en")
#print(llm.model)
#print(llm.model.config)
print(llm.model.config.is_encoder_decoder)
