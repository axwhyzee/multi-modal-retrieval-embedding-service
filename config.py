import os

from dotenv import find_dotenv, load_dotenv
from event_core.config import get_env_var

load_dotenv(find_dotenv())


EMBEDDING_DIM = 512

INDEX_NAME = "multi-modal-vectors"

TOP_N_MULTIPLIER = 5  # fetch top 5n cands, rerank and output top n ranked


def get_pinecone_api_key() -> str:
    return get_env_var("PINECONE_API_KEY")
