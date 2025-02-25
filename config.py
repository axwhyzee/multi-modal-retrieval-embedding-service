import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


EMBEDDING_DIM = 512

INDEX_NAME = "multi-modal-vectors"


def get_pinecone_api_key() -> str:
    return os.environ["PINECONE_API_KEY"]
