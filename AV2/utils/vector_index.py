import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

def build_vector_store(documents: List, persist_directory: str = None, chunk_size: int = 500, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs_split = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    if persist_directory:
        vectorstore = Chroma.from_documents(
            docs_split,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    else:
        vectorstore = Chroma.from_documents(
            docs_split,
            embedding=embeddings
        )

    return vectorstore
