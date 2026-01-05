import pandas as pd

# Frases usadas nos slides (mesmo vocabulário da matriz)
sentencas = [
    "gato gosta de mar",
    "gosta de peixe",
    "gosta de",
    "peixe mar"
]

# Ordem das linhas/colunas exatamente como na imagem
vocab = ["gato", "gosta", "de", "peixe", "mar"]

def matriz_coocorrencia_por_sentenca(sentencas, vocab):
    # matriz zerada
    M = pd.DataFrame(0, index=vocab, columns=vocab)

    for s in sentencas:
        tokens = s.lower().split()
        # conta coocorrência 1 vez por sentença (remove repetição dentro da mesma frase)
        tokens_unicos = list(dict.fromkeys(tokens))

        for w in tokens_unicos:
            if w not in M.index:
                continue
            for c in tokens_unicos:
                if c not in M.columns or c == w:
                    continue
                M.loc[w, c] += 1

    return M

df = matriz_coocorrencia_por_sentenca(sentencas, vocab)
print(df)
