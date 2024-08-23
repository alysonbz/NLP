from numpy.distutils.misc_util import clean_up_temporary_directory
from transformers import pipeline
llm = pipeline("summarization", model="facebook/bart-large-cnn")
long_text = """Copa do Nordeste: Another key competition where Ceará has shone is the Copa do Nordeste, a prestigious tournament featuring clubs from the northeastern region of Brazil. Ceará has won the Copa do Nordeste several times, with these victories adding significant prestige to the club's history. The tournament is highly competitive, and winning it is a source of immense pride for the fans and the region as a whole. The club's triumphs in this tournament underscore its status as a powerhouse in northeastern football."""
outputs = llm(long_text, max_length=60, clean_up_tokenization_spaces=True)
print(outputs[0]['summary_text'])