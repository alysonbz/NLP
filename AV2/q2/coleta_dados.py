import wikipedia

wikipedia.set_lang("pt")

times = ["São Paulo Futebol Clube", "Sport Club Corinthians Paulista", "Clube de Regatas do Flamengo"]

docs = {}

for t in times:
    docs[t] = wikipedia.page(t).content

for nome, texto in docs.items():
    with open(f"{nome}.txt", "w", encoding="utf-8") as f:
        f.write(texto)

