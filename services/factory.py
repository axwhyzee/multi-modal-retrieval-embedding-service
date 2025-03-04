from abc import ABC, abstractmethod
from typing import Dict, List, Type

from dependency_injector.wiring import Provide, inject
from event_core.domain.types import Modal

from adapters.embedder import AbstractEmbeddingModel
from bootstrap import DIContainer


class AbstractEmbedderFactory(ABC):
    @classmethod
    @abstractmethod
    @inject
    def embed(
        cls,
        data: bytes,
        emb_model: AbstractEmbeddingModel = Provide[DIContainer.emb_model],
    ) -> List[float]:
        raise NotImplementedError


class TextIndexer(AbstractEmbedderFactory):
    @classmethod
    @inject
    def embed(
        cls,
        data: bytes,
        emb_model: AbstractEmbeddingModel = Provide[DIContainer.emb_model],
    ) -> List[float]:
        return emb_model.embed_text(data.decode("utf-8"))


class ImageEmbedder(AbstractEmbedderFactory):
    @classmethod
    @inject
    def embed(
        cls,
        data: bytes,
        emb_model: AbstractEmbeddingModel = Provide[DIContainer.emb_model],
    ) -> List[float]:
        return emb_model.embed_image(data)


ModalToChunkEmbedder: Dict[Modal, Type[AbstractEmbedderFactory]] = {
    Modal.TEXT: TextIndexer,
    Modal.IMAGE: ImageEmbedder,
}
