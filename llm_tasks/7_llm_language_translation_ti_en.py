"""
Translation of Tigrinya to English using Hugging Face Transformers
"""

from transformers import pipeline
from transformers.utils import logging
import warnings
 
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

translator = pipeline(task="translation_ti_to_en", model="Helsinki-NLP/opus-mt-ti-en", device=0)

text = "እታ ከተማ ኣዝያ ጽብቅቲ እያ። ብ በዝሒ በጻሕቲ ኸኣ ምልእ ዝበለት እያ።"

output = translator(text, clean_up_tokenization_spaces=True)

print(output[0]["translation_text"])
