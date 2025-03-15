import logging
from abc import ABC, abstractmethod
from typing import ClassVar, List

logger = logging.getLogger(__name__)


class AbstractEmbeddingModel(ABC):

    EMBEDDING_DIM: ClassVar[int]

    @abstractmethod
    def embed(self, data: bytes) -> List[float]:
        raise NotImplementedError


class TextModel(AbstractEmbeddingModel): ...


class VisionModel(AbstractEmbeddingModel): ...


class PlotModel(AbstractEmbeddingModel): ...


class CodeModel(AbstractEmbeddingModel): ...
