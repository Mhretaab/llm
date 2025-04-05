from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

model_name = "papluca/xlm-roberta-base-language-detection"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

lang_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

text1 = "I am learning how to use Hugging Face Transformers for language identification."
# "እታ ከተማ ኣዝያ ጽብቅቲ እያ። ብ በዝሒ በጻሕቲ ኸኣ ምልእ ዝበለት እያ።"
result1 = lang_classifier(text1)
print(result1)

text2 = "እታ ከተማ ኣዝያ ጽብቅቲ እያ። ብ በዝሒ በጻሕቲ ኸኣ ምልእ ዝበለት እያ።"
result2 = lang_classifier(text2)
print(result2)