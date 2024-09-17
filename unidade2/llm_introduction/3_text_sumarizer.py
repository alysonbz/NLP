from transformers import pipeline

# Inicializa o pipeline de sumarização de texto
summarizer = pipeline("summarization")

# Texto para sumarizar
text = """
Artificial Intelligence (AI) is transforming the world at an unprecedented pace.
From healthcare to finance, AI applications are being integrated into everyday operations,
driving efficiency and enabling new possibilities. The advancements in AI technology
have led to breakthroughs in machine learning, natural language processing, and robotics.
However, the rapid growth also brings challenges, including ethical considerations,
job displacement, and the need for regulation. As AI continues to evolve, it is crucial
for society to address these challenges while harnessing the potential benefits
for a better future.
"""

# Realiza a sumarização do texto
summary = summarizer(text, max_length=60, min_length=25, do_sample=False)

# Exibe o resumo gerado
print(summary[0]['summary_text'])