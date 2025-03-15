import logging
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List

from pinecone import Pinecone, ServerlessSpec  # type: ignore
from pinecone.data.index import Index  # type: ignore
from pinecone.openapi_support.exceptions import PineconeApiException  # type: ignore

from config import get_pinecone_api_key

logger = logging.getLogger(__name__)


class AbstractVectorRepo(ABC):
    @abstractmethod
    def insert(
        self, index_name: str, namespace: str, key: str, vec: List[float]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(
        self, index_name: str, namespace: str, vec: List[float], top_k: int = 5
    ) -> List[str]:
        raise NotImplementedError


class PineconeRepo(AbstractVectorRepo):
    def __init__(
        self,
        index_names: Iterable[str],
        index_dims: Iterable[int],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._pc = Pinecone(api_key=get_pinecone_api_key())
        self._indexes: Dict[str, Index] = {
            name: self._get_or_create_index(name, dim)
            for name, dim in zip(map(str.lower, index_names), index_dims)
        }

    def _get_or_create_index(self, index_name: str, index_dim: int) -> Index:
        if not self._pc.has_index(index_name):
            logger.info(f"Creating index {index_name}")
            try:
                self._pc.create_index(
                    name=index_name,
                    dimension=index_dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            except PineconeApiException:
                # parallel process has created index
                pass
        return self._pc.Index(index_name)

    def insert(
        self, index_name: str, namespace: str, key: str, vec: List[float]
    ) -> None:
        index_name = index_name.lower()
        logger.info(f"Inserting {index_name=} {namespace=} {key=}")
        self._indexes[index_name].upsert(
            vectors=[{"id": key, "values": vec}], namespace=namespace
        )

    def query(
        self, index_name: str, namespace: str, vec: List[float], top_k: int = 5
    ) -> List[str]:
        index_name = index_name.lower()
        results = self._indexes[index_name].query(
            namespace=namespace,
            vector=vec,
            top_k=top_k,
            include_values=False,
            include_metadata=False,
        )
        return list(map(lambda match: match["id"], results["matches"]))
