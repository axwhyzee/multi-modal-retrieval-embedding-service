import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


# https://huggingface.co/openai/clip-vit-large-patch14-336
EMBEDDER_MODEL = "openai/clip-vit-large-patch14-336"

# https://huggingface.co/BAAI/bge-reranker-v2-m3
TEXT_TEXT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

EMBEDDING_DIM = 768

INDEX_NAME = "multi-modal-vectors"


def get_pinecone_api_key() -> str:
    return os.environ["PINECONE_API_KEY"]
