import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


# https://huggingface.co/openai/clip-vit-large-patch14-336
MODEL_PATH = "assets/clip-vit-large-patch14-336"

EMBEDDING_DIM = 768

INDEX_NAME = "multi-modal-vectors"


def get_pinecone_api_key() -> str:
    return os.environ["PINECONE_API_KEY"]
