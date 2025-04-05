"""
Translation of English to Spanish using Hugging Face Transformers
"""

from transformers import pipeline
from transformers.utils import logging
import warnings

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

translator = pipeline(task="translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es", device=0)

text = "Walking amid Gion's Machiya wooden houses was a mesmerizing experience."

output = translator(text, clean_up_tokenization_spaces=True)

print(output[0]["translation_text"])
