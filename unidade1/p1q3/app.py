import streamlit as st
import pandas as pd
from utils import stemmar, tokenizar, lemmatizar, corrigir, remover_stopwords

# 🎨 Configurações de página
st.set_page_config(page_title="Corretor de Texto", page_icon="📝", layout="centered")

# 💡 Estilo personalizado
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .stTextInput > div > div > input {
            border: 2px solid #4CAF50;
        }
        .titulo {
            text-align: center;
            color: #2E7D32;
            font-size: 2em;
            font-weight: bold;
        }
        .subtitulo {
            font-size: 1.2em;
            color: #555;
            margin-top: 1.5em;
        }
    </style>
""", unsafe_allow_html=True)

# 🧠 Título
st.markdown("<h1 class='titulo'>📝 Corretor de Texto</h1>", unsafe_allow_html=True)

st.markdown("Insira uma frase e veja o texto corrigido, tokens, lemas e stems!")

# ✍️ Entrada de texto
frase = st.text_input("Digite uma frase:", placeholder="Ex: Eu gostu di pyton e linguage natural")

# 📊 Processamento
if frase:
    with st.spinner("Processando..."):
        palavras_corrigidas = corrigir(frase)
        texto_corrigido = "".join(palavras_corrigidas)

        palavras_stem = stemmar(texto_corrigido)
        palavras_token = tokenizar(texto_corrigido)
        palavras_lemma = lemmatizar(texto_corrigido)
        palavras_sem_stopwords = remover_stopwords(texto_corrigido)

    # ✅ Resultado da correção
    st.markdown("<div class='subtitulo'>✅ Texto Corrigido:</div>", unsafe_allow_html=True)
    st.success(texto_corrigido)

    # 🧹 Remoção de stopwords
    st.markdown("<div class='subtitulo'>🧹 Texto sem Stopwords:</div>", unsafe_allow_html=True)
    st.info(" ".join(palavras_sem_stopwords))

    # 📋 Tokens, Lemmas e Stems
    st.markdown("<div class='subtitulo'>🔠 Análise de Palavras:</div>", unsafe_allow_html=True)
    df = pd.DataFrame({
        'Token': palavras_token,
        'Lemma': palavras_lemma,
        'Stem': palavras_stem,
    })
    st.dataframe(df, width='stretch')

else:
    st.warning("⚠️ Digite uma frase acima para começar!")
