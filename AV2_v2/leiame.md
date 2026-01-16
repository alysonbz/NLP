# AVALIAÇÃO 2
Orientações para execução da prova.

Esse documento exibe as descrições das questões e a relação dos datasets que devem ser utiizados pelos alunos e alunas.

O modelo de documento seguinte mostra como você deve registrar por escrito o desenvolvimento. https://docs.google.com/document/d/1hIwPx9W-k3LnXRJrkWYTsbrtx4NfP88_/edit?usp=sharing&ouid=118351454454462119994&rtpof=true&sd=true

Aluno - Dataset

ARIELLY GONCALVES LIMA: https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets

CARLOS EDUARDO TELES ALENCAR: https://www.kaggle.com/datasets/hrmello/brazilian-portuguese-hatespeech-dataset

FRANCISCA MARILIA DE OLIVEIRA RODRIGUES: https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets?select=buscape.csv

JOSE MARIO OLIVEIRA PATRICIO: https://www.kaggle.com/datasets/moesiof/portuguese-narrative-essays

LEANDRO NASCIMENTO ADEGAS : https://github.com/kamplus/FakeNewsSetGen/tree/master/Dataset

PEDRO VINICIUS FELIX ROSA VIANA: https://www.kaggle.com/datasets/brunoluvizotto/brazilian-headlines-sentiments

VICTOR MATHEUS ARAUJO OLIVEIRA: https://huggingface.co/datasets/nilc-nlp/assin

---

## Questão 1
Utilizando o melhor caso como referencia da AV1, compare agora o processo de classificação de texto do melhor algoritmo da AV1 com uma LLM de sua preferencia. Gere resultados e mostre a avaliação qualitativa e quantitativa.


## Questão 2
Escolha ou crie sinteticamente 3 arquivos para aplicação de RAG. Elabore todo o pipeline, avalie e justifique o motivo da escolha do pipeline escolhido.

---

### **Observações para o Relatório**
No relatório, organize a comparação de resultados entre os modelos, capture as informações de resultados da AV1 para enriquecer a comparação.

Discutir organizadamente os resultados obtidos de cada questão. Ao concluir o relatório, compartilhar com **alysonbnr@ufc.br** até **12-01**.

### **Observações para a Apresentação**
Criar apresentação para realizar até 12-01.


---

---

## Fluxo do projeto

### A pasta `rag_docs/` é a pasta onde contém o 'conhecimento' da RAG/chat

Este projeto implementa, de ponta a ponta, duas soluções complementares para Processamento de Linguagem Natural aplicadas a notícias em português.

Na **Questão 1**, o trabalho constrói um pipeline clássico de **classificação supervisionada** para detectar *fake news* a partir de **títulos**. O dataset é carregado, rotulado (Fake = 1, Real = 0), embaralhado e dividido de forma estratificada em treino e teste. Em seguida, os textos são convertidos em vetores usando **TF-IDF com unigramas e bigramas**, e diferentes modelos tradicionais (ex.: Regressão Logística, SVM linear, Naive Bayes) são comparados por validação cruzada. O melhor modelo é avaliado no conjunto de teste com métricas quantitativas (accuracy, precision, recall, F1 e matriz de confusão) e também com análise qualitativa dos erros.

Ainda na **Questão 1**, esse melhor classificador é comparado com uma **LLM open source rodando localmente via Ollama**, que recebe cada título por prompt e retorna uma decisão “Fake/Real” (com justificativa e confiança). A avaliação repete as mesmas métricas do classificador clássico e inclui uma análise qualitativa de casos em que a LLM e o modelo clássico divergem, destacando vantagens e limitações práticas (custo, reprodutibilidade, estabilidade, dependência do prompt).

Na **Questão 2**, o projeto implementa um pipeline de **RAG (Retrieval-Augmented Generation)** com três documentos (sintéticos ou substituíveis por arquivos reais). Os textos são divididos em chunks com overlap, indexados com **TF-IDF** e recuperados por **similaridade cosseno**, retornando os Top-K trechos mais relevantes para cada pergunta. Esses trechos são inseridos no prompt e enviados à **LLM local (Ollama)**, que é instruída a responder apenas com base no contexto recuperado e recusar quando não há evidência suficiente. O retriever é avaliado quantitativamente com **hit@k**, e a geração é avaliada qualitativamente pela fidelidade ao contexto (redução de “alucinação”).

> Em resumo, o projeto entrega um pacote simples, reprodutível e gratuito (open source) que cobre:\
> **(1) baseline clássico forte para classificação de texto**,\
> **(2) comparação direta com uma LLM**, e\
> **(3) uma prova de conceito RAG completa, com retrieval + geração fundamentada**, tudo com códigos em Python focados em clareza e documentação para preenchimento do relatório.


---

# AV2 – NLP: Classificação de Fake News + RAG com LLM Open Source (Ollama)

Este projeto entrega uma solução completa para a **AV2** com duas partes:

1. **Questão 1 (Classificação de Texto):** compara um **modelo clássico supervisionado** (TF-IDF + algoritmo de ML) com uma **LLM open source** rodando localmente via **Ollama**.
2. **Questão 2 (RAG):** implementa um pipeline **RAG (Retrieval-Augmented Generation)** com **3 documentos**, indexação vetorial simples (TF-IDF) e geração condicionada ao contexto recuperado usando **Ollama**.

O foco é manter tudo **simples, gratuito e reprodutível**, sem `argparse` e sem `def main()`.

---

## Visão geral do fluxo

### Questão 1 – Comparação Clássico vs LLM
**Entrada:** títulos de notícias (Fake/Real)  
**Saída:** métricas quantitativas (accuracy, precision, recall, F1, matriz de confusão) e análise qualitativa (erros e discordâncias)

**Fluxo:**
1. Carrega dois CSVs: um com notícias Fake e outro com notícias Real.
2. Cria rótulos (Fake=1, Real=0), concatena e embaralha.
3. Separa treino/teste com split estratificado.
4. Constrói embedding **TF-IDF** (unigramas + bigramas).
5. Treina e compara múltiplos classificadores clássicos (LogReg, LinearSVC, NB, SGD, RF) por validação cruzada.
6. Seleciona o melhor modelo por **F1**.
7. Avalia no teste (quantitativo + matriz de confusão) e imprime exemplos de erros (qualitativo).
8. Em paralelo, aplica uma **LLM via Ollama** para classificar os mesmos títulos por prompt e mede as mesmas métricas.
9. Mostra casos de discordância entre clássico e LLM para discussão qualitativa.

---

### Questão 2 – RAG (Retrieval-Augmented Generation)
**Entrada:** três documentos (arquivos) + pergunta do usuário  
**Saída:** resposta gerada pela LLM **fundamentada** nos trechos recuperados (Top-K), além de avaliação do retriever via **hit@k**

**Fluxo:**
1. Cria (ou lê) 3 arquivos de texto.
2. Divide os documentos em **chunks** (com overlap).
3. Indexa os chunks em um “banco vetorial” simples (matriz TF-IDF).
4. Dada uma pergunta:
   - gera vetor TF-IDF da pergunta
   - calcula similaridade cosseno com todos os chunks
   - recupera Top-K chunks mais relevantes
5. Monta um prompt com:
   - regra “responda apenas com base no contexto”
   - contexto = chunks recuperados
   - pergunta do usuário
6. Envia para a LLM via Ollama e imprime a resposta.
7. Mede **hit@k** com perguntas de teste (quantitativo) e avalia fidelidade/resposta com/sem evidência (qualitativo).

---

## Embedding e banco vetorial (para o relatório)

### Embedding utilizado
- **TF-IDF (TfidfVectorizer)**: transforma texto em **vetores esparsos**, onde cada dimensão corresponde a um termo/n-gram do vocabulário.
- Configuração típica:
  - `ngram_range=(1,2)` (unigramas e bigramas)
  - `min_df` e `max_df` ajustados conforme o caso (classificação vs RAG)

**Por que TF-IDF?**
- É excelente para textos curtos (títulos).
- É interpretável e fácil de justificar em relatório.
- Não exige GPU, não exige API de embeddings e roda muito rápido.

### “Banco vetorial” utilizado
- **Índice em memória** composto por:
  - uma **matriz esparsa** de vetores TF-IDF dos chunks (`X`)
  - metadados por chunk (`doc`, `chunk_id`)
- Recuperação por **similaridade cosseno** e seleção Top-K.

**Por que esse banco vetorial simples?**
- O corpus do RAG é pequeno (3 documentos); uma base vetorial completa (Milvus/Weaviate) seria complexidade desnecessária.
- Permite auditoria fácil: você vê exatamente quais chunks foram recuperados e com quais scores.

---

## Requisitos

### Python (bibliotecas)
Instale as dependências:


`pip install pandas scikit-learn matplotlib requests`


---

```
--- Top recuperados (retrieval) ---
- resumo_dataset.txt | chunk 0 | score=0.524
- resumo_dataset.txt | chunk 1 | score=0.492
- q1_relatorio.txt | chunk 0 | score=0.486
- relatorio_base.txt | chunk 2 | score=0.459
```

### Cada linha significa:

* `resumo_dataset.txt:` o arquivo de onde veio o chunk.

* `chunk 0 / 1 / 2:` índice do chunk dentro daquele arquivo (começa em 0).

* `score=...:` similaridade (cosseno) entre o embedding da pergunta e o embedding do chunk.

### Interpretação prática do score:

* quanto maior, mais “próximo semanticamente” da pergunta.

* nesse seu caso, o sistema está dizendo: “os trechos mais relevantes parecem estar em `resumo_dataset.txt` (chunks 0 e 1)”.


---

---

# DEPOIS DAS ATUALIZAÇÕES:

A seguir está uma explicação completa e organizada do seu projeto (AV2 – NLP), cobrindo objetivo, Questão 1, Questão 2 (RAG), API/LLM, embeddings, “banco vetorial” e observações relevantes para relatório.

---

## 1) Intuito do projeto

O projeto tem dois objetivos principais:

1. **Comparar abordagens de NLP clássicas vs LLM** para a tarefa de **classificação binária de notícias** (fake vs real) usando **títulos em português**.
2. Implementar um **pipeline de RAG (Retrieval-Augmented Generation)** com chat interativo, onde a LLM responde **somente com base** em documentos locais (`.txt`), recuperando trechos relevantes via embeddings.

Em termos didáticos, a AV2 demonstra dois paradigmas:

* **NLP supervisionado clássico** (features + classificador)
* **LLM/RAG** (embeddings + recuperação + geração)

---

## 2) Questão 1 (Q1) – Como é feito o treinamento

### 2.1 Dados

* O script lê dois arquivos:

  * `News_fake.csv` (rotulado como **FAKE = 1**)
  * `News_notFake.csv` (rotulado como **REAL = 0**)
* Une os datasets, embaralha e faz divisão **treino/teste** (estratificada).

### 2.2 Pré-processamento / Representação (features)

A representação do texto no modelo clássico é feita com **TF-IDF**:

* `TfidfVectorizer`
* `ngram_range=(1,2)` (unigramas e bigramas)
* `min_df=2`
* `max_df=0.95`

Ou seja: cada título vira um vetor esparso (bag-of-ngrams ponderado por TF-IDF).

### 2.3 Modelos treinados e seleção

Você treina e compara múltiplos classificadores clássicos, como:

* **Logistic Regression**
* **LinearSVC**
* **MultinomialNB**
* **SGDClassifier**
* **RandomForest**

A seleção do “melhor” é feita por validação cruzada (tipicamente 5-fold) usando métrica de desempenho (no seu caso, F1 é a principal).

### 2.4 Avaliação final (teste)

Após escolher o melhor modelo no treino/CV, você avalia no conjunto de teste com:

* Accuracy
* Precision / Recall / F1
* Classification report
* Matriz de confusão

### 2.5 Comparação com LLM (Q1)

Além do baseline clássico, você também roda uma classificação via **LLM (Ollama)**, por prompt, para os mesmos títulos e calcula as mesmas métricas. Isso permite comparar:

* “modelo tradicional supervisionado” vs “LLM zero-shot/few-shot”.

---

## 3) Questão 2 (Q2) – Como é feito o RAG (chat interativo)

A Q2 implementa um pipeline de **RAG** para responder perguntas com base em documentos `.txt` locais.

### 3.1 Ingestão

* Lê todos os arquivos `.txt` de uma pasta (por padrão `./rag_docs`).

### 3.2 Chunking

Cada documento é dividido em partes menores (chunks) por janela deslizante:

* `CHUNK_SIZE = 900` **caracteres**
* `CHUNK_OVERLAP = 120` **caracteres**

Isso evita perda de contexto na borda do corte (overlap).

### 3.3 Indexação (embeddings)

Para cada chunk, você gera um embedding denso via Ollama usando o modelo de embeddings configurado (default `bge-m3`).

### 3.4 Banco vetorial / Índice

Você tem dois cenários:

* **Antes (implementação original):** índice em memória com NumPy

  * matriz `E` (embeddings) + normalização `E_norm`
  * busca por cosseno via `E_norm @ q`

* **Agora (versão integrada com ChromaDB):** índice persistente no **ChromaDB**

  * `collection.upsert(...)` grava:

    * `documents` (texto chunk)
    * `embeddings` (vetores)
    * `metadatas` (`doc`, `chunk_id`)
    * `ids` (IDs estáveis)
  * `collection.query(...)` recupera os Top-K

Em ambos os casos, o conceito é o mesmo: você monta um **índice vetorial** para fazer similaridade.

### 3.5 Recuperação (retrieval)

Quando o usuário pergunta algo:

1. gera embedding da pergunta
2. busca os **Top-K** chunks mais similares (`TOP_K=4`)
3. retorna os trechos com metadados

### 3.6 Geração (LLM)

Com os chunks recuperados, você monta o prompt:

* insere o “CONTEXTO” com `[doc | chunk | score] + texto`
* força regra: “responda somente com base no contexto”
* se não houver evidência: responder exatamente
  **"Não há informação suficiente no contexto."**

Então chama o endpoint de geração do Ollama e imprime a resposta.

---

## 4) Qual API está sendo utilizada

Você utiliza a **API HTTP local do Ollama**, rodando em:

* `OLLAMA_BASE_URL` (default `http://localhost:11434`)

Endpoints principais:

* **Geração (LLM):** `POST /api/generate`
* **Embeddings:** `POST /api/embed`
* (checar saúde/lista): `GET /api/tags`

Isso significa que toda inferência e embedding ocorre localmente no seu PC, sem API externa paga.

---

## 5) Qual banco vetorial está sendo utilizado

Na sua versão atual com integração:

* **ChromaDB (persistente local)**, com `PersistentClient(path=".chroma_db")`
* Coleção: por padrão `rag_docs`

Na versão original (antes do Chroma), não havia um “banco vetorial” dedicado; era um índice manual em NumPy. Com o Chroma, você passa a ter um banco vetorial reconhecido e persistente.

---

## 6) Qual embedding está sendo utilizado

Para Q2 (RAG), o embedding utilizado é:

* **`bge-m3`** (default), via variável `OLLAMA_EMBED_MODEL`

Ou seja:

* chunks → embedding `bge-m3`
* pergunta → embedding `bge-m3`
* busca por similaridade no espaço vetorial

Para Q1 (clássico), a “representação vetorial” é TF-IDF (não é embedding neural; é vetorização estatística).

---

## 7) Outras explicações relevantes (para enriquecer seu relatório)

### 7.1 Por que Q1 funciona bem com títulos

* Títulos têm padrões lexicais fortes.
* TF-IDF + linear models (LogReg/LinearSVC) costuma ser excelente em classificação de texto curto.

### 7.2 Por que Q2 (RAG) é necessário

* A LLM sozinha pode “alucinar”.
* O RAG impõe *grounding*: a resposta deve vir de trechos recuperados.
* Isso aproxima o sistema de um “assistente” baseado em base de conhecimento local.

### 7.3 Trade-offs do chunking

* Chunk maior: mais contexto, mas mais ruído no retrieval
* Chunk menor: mais precisão, mas pode perder explicações longas
* Seu `900` caracteres com overlap `120` é um valor razoável para textos curtos/médios; e com `TOP_K=4` tende a caber bem no contexto.

### 7.4 Persistência e reindexação (com Chroma)

* Você calcula um **fingerprint** do corpus + parâmetros (chunk, overlap, embedding model).
* Se mudar qualquer coisa, você reindexa para não misturar vetores de corpora diferentes.

### 7.5 Métricas e avaliação (o que dá para citar)

* Q1: métricas supervisionadas padrão (F1/acc/confusion matrix)
* Q2: dá para avaliar o retriever com:

  * “hit@k” (se o chunk certo aparece no Top-K)
  * avaliação qualitativa de groundedness (se a resposta cita e respeita o contexto)

---

Se você quiser, eu também posso te entregar um texto já “em formato de relatório” (Introdução, Metodologia, Implementação, Resultados, Discussão) com base exatamente no que o projeto faz, pronto para colar no seu PDF final.







