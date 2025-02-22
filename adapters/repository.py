import logging
from abc import ABC, abstractmethod
from typing import List

from pinecone import Pinecone, ServerlessSpec  # type: ignore

from config import EMBEDDING_DIM, INDEX_NAME, get_pinecone_api_key

logger = logging.getLogger(__name__)


class AbstractVectorRepo(ABC):
    @abstractmethod
    def insert(self, key: str, vec: List[float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, user: str, vec: List[float], top_k: int = 5) -> List[str]:
        raise NotImplementedError


class PineconeRepo(AbstractVectorRepo):
    def __init__(self):
        pc = Pinecone(api_key=get_pinecone_api_key())

        if not pc.has_index(INDEX_NAME):
            logger.info(f"Creating index {INDEX_NAME}")
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self._index = pc.Index(INDEX_NAME)

    def insert(self, key: str, vec: List[float]) -> None:
        user = key.split("/")[0]
        logger.info(f"Inserting {key=} for {user=}")
        self._index.upsert(
            vectors=[{"id": key, "values": vec}], namespace=user
        )

    def query(self, user: str, vec: List[float], top_k: int = 5) -> List[str]:
        results = self._index.query(
            namespace=user,
            vector=vec,
            top_k=top_k,
            include_values=False,
            include_metadata=False,
        )
        return list(map(lambda match: match["id"], results["matches"]))
