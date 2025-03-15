FROM python:3.11-slim

COPY . .

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

RUN huggingface-cli download openai/clip-vit-base-patch32

RUN huggingface-cli download google/deplot

RUN huggingface-cli download BAAI/bge-reranker-v2-m3

RUN huggingface-cli download microsoft/unixcoder-base

# RUN huggingface-cli download vidore/colpali-v1.2

# RUN huggingface-cli download vidore/colpaligemma-3b-pt-448-base

EXPOSE 5000
