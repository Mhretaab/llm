"""
Using a pipeline for summarization
Run a summarization pipeline using the "facebook/bart-large-cnn" model from the Hugging Face hub.
"""

from transformers import pipeline

summarizer = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn",
    device=0  # Use GPU if available, otherwise use CPU
)

text = """Walking amid Gion's Machiya wooden houses is a mesmerizing experience. The
beautifully preserved structures exuded an old-world charm that transports visitors
back in time, making them feel like they had stepped into a living museum. The glow of
lanterns lining the narrow streets add to the enchanting ambiance, making each stroll a
memorable journey through Japan's rich cultural history."""

summary = summarizer(text, max_length=50)
print(summary[0]['summary_text'])