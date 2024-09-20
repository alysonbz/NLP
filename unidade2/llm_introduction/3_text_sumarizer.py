from numpy.distutils.misc_util import clean_up_temporary_directory
from transformers import pipeline
llm = pipeline("summarization", model="facebook/bart-large-cnn")
long_text = """I said: Remember this moment
In the back of my mind
The time we stood with our shaking hands
The crowds in stands went wild
We were the kings and the queens
And they read off our names
The night you danced like you knew our lives
Would never be the same"""
outputs = llm(long_text, max_length=60, clean_up_tokenization_spaces=True)
print(outputs[0]['summary_text'])