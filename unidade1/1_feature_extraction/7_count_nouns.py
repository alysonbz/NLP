import spacy

nlp = spacy.load('en_core_web_sm')


# Retorna o número de substantivos próprios
def proper_nouns(text, model=nlp):
    # Criar objeto doc
    doc = model(text)

    # Gerar lista de tags POS
    pos = [token.pos_ for token in doc]

    # Contar o número de substantivos próprios
    proper_noun_count = pos.count('PROPN')

    return proper_noun_count


# Exemplo de uso da função
print(proper_nouns("Abdul, Bill and Cathy went to the market to buy apples.", nlp))
