# 1) Apresentação (em PT) sobre Stanza (alternativa ao spaCy)
# Título: “Stanza: Pipeline NLP da Stanford para PT-BR”
#
# O que é: Biblioteca de PLN com modelos neurais (PyTorch) treinados no UD. Faz tokenização, sentencização, POS, lemas, dependências e NER.
"""
Questão 1 – Demonstração de outra biblioteca de PLN: Stanza (Stanford NLP)
Mostra tokenização, POS tagging e lematização em português.

biblioteca criada pela Universidade de Stanford para Processamento de Linguagem Natural

baseada em redes neurais treinadas no padrão Universal Dependencies (UD)

Diferente do spaCy, o Stanza vem com um pipeline pronto
"""

import stanza

# Baixar modelo de português (só na 1ª execução)
# stanza.download('pt')

# Criar o pipeline completo em português
nlp = stanza.Pipeline('pt')

# Texto de exemplo
texto = "O rato roeu a roupa do rei de Roma."

# Processar o texto
doc = nlp(texto)

print("=== Texto processado ===")
print(texto)

print("\n=== Tabela: Token | POS | Lemma ===")
for sent in doc.sentences:
    for w in sent.words:
        print(f"{w.text:10} | {w.upos:6} | {w.lemma}")