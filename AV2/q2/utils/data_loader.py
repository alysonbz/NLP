import os
from langchain_core.documents import Document
from typing import List

def load_txt_folder(folder_path: str, encoding: str = "utf8") -> List[Document]:
    """Carrega todos os arquivos .txt de uma pasta como documentos LangChain."""
    docs = []

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".txt"):
            path = os.path.join(folder_path, fname)
            print("Carregando:", path)

            with open(path, "r", encoding=encoding) as f:
                text = f.read()

            if not text.strip():
                continue

            metadata = {
                "source": fname,
                "folder": os.path.basename(folder_path)
            }

            docs.append(Document(page_content=text, metadata=metadata))

    return docs
