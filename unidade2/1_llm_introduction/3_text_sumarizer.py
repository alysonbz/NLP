from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization")

text = """
Artificial Intelligence is a field of computer science that aims to create
systems capable of performing tasks that normally require human intelligence,
such as reasoning, learning, and decision-making.
"""

summary = summarizer(text, max_length=40)

print(summary[0]["summary_text"])
