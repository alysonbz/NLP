# No arquivo 3_text_sumarizer.py elabore uma sumarização de texto utilizando
# o modelo default do módulo pipeline

from transformers import pipeline

llm = pipeline("summarization", model="facebook/bart-large-cnn")
long_text = (""" Eu era uma simples menina e tudo ia bem e uma princesinha me tornei, 
mas agora o que fazer? Quero ir além, há muito pra aprender. Tenho um castelo agora, meu novo lar
e na escola tenho amigos pra brincar um mundo novo e eu quero acreditar e eu vou aproveitar. 
(Princesinha Sofia) Quero saber e num castelo descobrir
(Princesinha Sofia) Mil aventuras e aonde eu devo ir
(Sofia) Princesa quero ser, é tudo que eu vou ser
Princesinha Sofia!""")

outputs = llm(long_text, max_length=60, clean_up_tokenization_spaces=True)
print(outputs[0]['summary_text'])