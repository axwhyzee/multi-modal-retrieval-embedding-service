import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import List

from pinecone import Pinecone, ServerlessSpec  # type: ignore

from config import EMBEDDING_DIM, INDEX_NAME, get_pinecone_api_key

logger = logging.getLogger(__name__)


class Modality(StrEnum):
    IMAGE = "IMAGE"
    TEXT = "TEXT"
    VIDEO = "VIDEO"


class AbstractVectorRepo(ABC):
    @abstractmethod
    def insert(self, namespace: str, key: str, vec: List[float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(
        self, namespace: str, vec: List[float], top_k: int = 5
    ) -> List[str]:
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

    def insert(self, namespace: str, key: str, vec: List[float]) -> None:
        logger.info(f"Inserting {key=} in {namespace=}")
        self._index.upsert(
            vectors=[{"id": key, "values": vec}], namespace=namespace
        )

    def query(
        self, namespace: str, vec: List[float], top_k: int = 5
    ) -> List[str]:
        results = self._index.query(
            namespace=namespace,
            vector=vec,
            top_k=top_k,
            include_values=False,
            include_metadata=False,
        )
        return list(map(lambda match: match["id"], results["matches"]))
