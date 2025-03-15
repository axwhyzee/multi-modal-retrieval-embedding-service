import logging
from abc import ABC, abstractmethod
from typing import List

logger = logging.getLogger(__name__)


class AbstractEmbeddingModel(ABC):

    @abstractmethod
    def embed(self, data: bytes) -> List[float]:
        raise NotImplementedError


class TextModel(AbstractEmbeddingModel): ...


class VisionModel(AbstractEmbeddingModel): ...


class PlotModel(AbstractEmbeddingModel): ...
