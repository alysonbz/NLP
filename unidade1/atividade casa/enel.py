from enelvo.normaliser import Normaliser
from enelvo.preprocessing import sanitize

norm = Normaliser(tokenizer='readable')

msg = 'Até hj vc n me respondeu. Oq aconteceu?'
resposta = norm.normalise(msg)
print(resposta)

print("_______________________________")

cap_pns = Normaliser(capitalize_pns=True)

cap_acs = Normaliser(capitalize_acs=True)

cacps_inis = Normaliser(capitalize_inis=True)

sanitizer = Normaliser(sanitize=True)


normalizador = Normaliser(tokenizer='readable', capitalize_inis=True, capitalize_pns=True, capitalize_acs=True, sanitize=True)

resposta = normalizador.normalise(msg)

print(resposta)
