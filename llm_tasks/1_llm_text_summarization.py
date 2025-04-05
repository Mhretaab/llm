"""
Using a pipeline for summarization
Run a summarization pipeline using the "cnicu/t5-small-booksum" model from the Hugging Face hub.

A long_text about the Eiffel Tower has been provided and the pipeline module from transformers is already imported.

Load the model pipeline for a summarization task using the model "cnicu/t5-small-booksum".
Generate the output by passing the long_text to the pipeline; limit the output to 50 tokens.
Access and print the summarized text only from the output
"""

from transformers import pipeline

# Load the model pipeline
summarizer = pipeline(task="summarization", model="cnicu/t5-small-booksum", device=0)

long_text = """
The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.
"""

# Pass the long text to the model
output = summarizer(long_text,  max_length=50, clean_up_tokenization_spaces=True, truncation=True)

# Access and print the summarized text
print(output[0]['summary_text'])