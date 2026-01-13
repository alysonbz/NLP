import os
import json
from langchain_core.documents import Document
from typing import List

def load_jsonl_folder(folder_path: str, encoding: str = "utf8") -> List[Document]:
    """Carrega todos os arquivos .jsonl de uma pasta como documentos LangChain."""
    docs = []

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".jsonl"):
            path = os.path.join(folder_path, fname)
            print("Carregando:", path)

            with open(path, "r", encoding=encoding) as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)

                    # Permite tanto {"text": "..."} quanto {"content": "..."}
                    text = (
                        data.get("text")
                    )

                    if text is None:
                        continue

                    metadata = data.get("metadata", {})
                    metadata["source"] = fname

                    docs.append(Document(page_content=text, metadata=metadata))

    return docs