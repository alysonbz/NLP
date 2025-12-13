from transformers import pipeline

# Load the summarization pipeline (modelo default)
summarizer = pipeline("summarization")

text = "Artificial intelligence is a field of computer science that focuses on creating systems capable of performing tasks that normally require human intelligence, such as learning, reasoning, and problem-solving."

summary = summarizer(text, max_length=40, min_length=10, do_sample=False)
print(summary[0]["summary_text"])
