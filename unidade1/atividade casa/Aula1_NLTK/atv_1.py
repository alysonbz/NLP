
# 1. Fazer uma apresentação para a aula sobre a usabilidade de outra biblioteca usando língua portuguesa.

"""
Flair
 > Biblioteca focada em NLP baseado em embeddings contextuais
 > Suporte direto ao português (POS tagging, NER)
 > Fácil de usar para tarefas práticas de etiquetagem, reconhecimento de entidades, análise de sentimentos, marcação de classes gramaticais, apoio especial para textos biomédicos, desambiguação e classificação de sentidos
 > Estrutura baseada diretamente no TorchPy, tornando mais fácil o treinamento
"""

from flair.data import Sentence
from flair.models import SequenceTagger

sent = Sentence("O presidente falou em Brasília no Senado Federal.")
tagger = SequenceTagger.load("flair/ner-multi")
tagger.predict(sent)
print(sent.to_tagged_string())
