from transformers import pipeline

# Definir o modelo de sumarização
llm = pipeline("summarization", model="facebook/bart-large-cnn")

# Novo texto longo
long_text = """A tecnologia tem transformado rapidamente o mundo do trabalho, impactando a forma como as empresas operam e os profissionais realizam suas atividades. A automação, inteligência artificial e big data são exemplos de inovações que têm revolucionado setores inteiros, desde a produção industrial até o atendimento ao cliente. Ao mesmo tempo, essas mudanças exigem que os trabalhadores desenvolvam novas habilidades, adaptando-se às demandas de um mercado em constante evolução. A educação e o treinamento profissional desempenham um papel crucial nesse cenário, garantindo que a força de trabalho esteja preparada para os desafios futuros."""

# Gerar a sumarização
outputs = llm(long_text, max_length=60, clean_up_tokenization_spaces=True)

# Imprimir a sumarização
print(outputs[0]['summary_text'])
