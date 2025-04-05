from transformers import pipeline
from transformers.utils import logging
import warnings

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

#generator = pipeline("text-generation", model="luel/gpt2-tigrinya-small", device=0)
generator = pipeline("text-generation", model="luel/gpt2-tigrinya-medium", device=0)

prompt = "ክልል ትግራይ"

output = generator(
    prompt, max_length=100,
    pad_token_id=generator.tokenizer.eos_token_id,
    truncation=True,
)

print(output[0]['generated_text'])
