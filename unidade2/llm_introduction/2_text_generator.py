#No arquivo 2_text_generator.py elabore uma geração de texto
# utilizando o modelo default do módulo pipeline
from transformers import pipeline
llm = pipeline("text-generation")
prompt = "Bonjou, Je m'appelle Maria"
output = llm(prompt, max_length=25)
print(output[0]['generated_text'])