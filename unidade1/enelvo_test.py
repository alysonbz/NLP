from enelvo.normaliser import Normaliser

norm = Normaliser(tokenizer='readable')

msg = "Até hj vc n me respondeu. Oq aconteceu?"

resposta = norm.normalise(msg)
print(resposta)

normalisador = Normaliser(tokenizer='readable', capitalize_inis=True,
                          capitalize_pns=True, capitalize_acs=True,
                          sanitize=True)

msg = "a maria foi ao shopp pq estava trsite, acho q hj foi uma dia dificiu"

resposta = normalisador.normalise(msg)
print(resposta)
