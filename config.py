import torch
from dotenv import find_dotenv, load_dotenv
from event_core.config import get_env_var

load_dotenv(find_dotenv())


TOP_N_MULTIPLIER = 3  # fetch top 3n cands, rerank and output top n ranked


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.xpu.is_available():
    DEVICE = torch.device("xpu")
else:
    DEVICE = torch.device("cpu")


def get_pinecone_api_key() -> str:
    return get_env_var("PINECONE_API_KEY")
