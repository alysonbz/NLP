import os
from typing import List, Optional, Union
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Vector store backends
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def build_vector_store(
    documents: List[Document],
    embedding_type: str = "openai",
    vectorstore_type: str = "chroma",
    persist_directory: Optional[str] = None,
    chunk_size: int = 200,
    chunk_overlap: int = 50,
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    collection_name: Optional[str] = None,
):
    # 1. Split docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs_split = splitter.split_documents(documents)

    # 2. Choose embeddings
    embedding_type = embedding_type.lower()
    if embedding_type == "openai":
        embeddings = OpenAIEmbeddings()
    elif embedding_type == "huggingface":
        embeddings = HuggingFaceEmbeddings(model_name=hf_model_name)
    elif embedding_type == "google":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",
                                                  api_key=GOOGLE_API_KEY,
                                                  task_type="QUESTION_ANSWERING")
    else:
        raise ValueError(f"Embedding type '{embedding_type}' não suportado")

    # 3. Choose vectorstore
    vectorstore_type = vectorstore_type.lower()
    if vectorstore_type == "chroma":
        if persist_directory:
            vs = Chroma.from_documents(docs_split, embedding=embeddings, persist_directory=persist_directory, collection_name=collection_name)
        else:
            vs = Chroma.from_documents(docs_split, embedding=embeddings, collection_name=collection_name)
    elif vectorstore_type == "faiss":
        vs = FAISS.from_documents(docs_split, embedding=embeddings)
    # elif vectorstore_type == "pgvector":
    #     vs = PGVector.from_documents(docs_split, embedding=embeddings, **kwargs.get("pgvector", {}))
    else:
        raise ValueError(f"Vectorstore type '{vectorstore_type}' não suportado")

    return vs
