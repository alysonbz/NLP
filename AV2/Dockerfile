FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
ENV PIP_DEFAULT_TIMEOUT=100
COPY requirements.txt .
# Instala a versão leve do torch
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "questao_2_rag.py"]