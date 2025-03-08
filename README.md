# Multi-Modal Retrieval Embedding Service
Embedding Service indexes uploaded objects into their respective `{USER}/{MODAL}` namespace, and yields objects most relevant to a provided text query.

## Setup
Setup Python env
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Download embedding and reranker models using `huggingface-cli` which comes installed with the `transformers` library
```
huggingface-cli download openai/clip-vit-base-patch32
huggingface-cli download BAAI/bge-reranker-v2-m3
huggingface-cli download vidore/colpali-v1.2
```

## Run
```
# app is required only by frontend to make queries
PYTHONPATH=. python entrypoints/app.py

# event consumer should be started when expecting doc uploads
PYTHONPATH=. python entrypoints/event_consumer.py
```
