# Usado para normalizar o texto, ou seja, corrigir, gírias, erros ortográficos, ...

# pip install enelvo

from enelvo.normaliser import Normaliser

norm = Normaliser(tokenizer='readable')
msg = 'Até hj vc n me respondeu. Oq aconteceu?'
resposta = norm.normalise(msg)
print(resposta)

# capitaliza nomes próprios
cap_pns = Normaliser(capitalize_pns = True)

# capitaliza acrônimos
cap_acs = Normaliser(capitalize_acs = True)

# capitaliza começos de frases
caps_inis = Normaliser(capitalize_inis = True)

# remove pontuações e emojis
sanitizer = Normaliser(sanitize = True)

# EXEMPLO
normalizador = Normaliser(tokenizer='readable', capitalize_pns = True, capitalize_acs = True, capitalize_inis = True, sanitize = True)

msg2 = 'a maria foi ao shopp pq estava trsite, acho q hj foi uma dia dificiu'
resposta2 = normalizador.normalise(msg2)
print('\n', resposta2)