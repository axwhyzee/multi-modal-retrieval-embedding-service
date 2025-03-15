import logging
from abc import ABC, abstractmethod
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)

TOP_K = 5


class AbstractReranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, candidates: Iterator[bytes], top_k: int = TOP_K
    ) -> Iterable[int]:
        raise NotImplementedError
