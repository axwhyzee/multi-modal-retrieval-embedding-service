from abc import ABC, abstractmethod
from typing import Dict, List, Type

from dependency_injector.wiring import Provide, inject

from adapters.embedder import AbstractEmbeddingModel
from adapters.types import FileExt, Modal
from bootstrap import DIContainer


class AbstractEmbedderFactory(ABC):
    @classmethod
    @abstractmethod
    @inject
    def embed(
        cls,
        obj: bytes,
        emb_model: AbstractEmbeddingModel = Provide[DIContainer.emb_model],
    ) -> List[float]:
        raise NotImplementedError


class TextIndexer(AbstractEmbedderFactory):
    @classmethod
    @inject
    def embed(
        cls,
        obj: bytes,
        emb_model: AbstractEmbeddingModel = Provide[DIContainer.emb_model],
    ) -> List[float]:
        return emb_model.embed_text(obj.decode("utf-8"))


class ImageEmbedder(AbstractEmbedderFactory):
    @classmethod
    @inject
    def embed(
        cls,
        obj: bytes,
        emb_model: AbstractEmbeddingModel = Provide[DIContainer.emb_model],
    ) -> List[float]:
        return emb_model.embed_image(obj)


ModalToChunkEmbedder: Dict[Modal, Type[AbstractEmbedderFactory]] = {
    Modal.TEXT: TextIndexer,
    Modal.IMAGE: ImageEmbedder,
    Modal.VIDEO: ImageEmbedder,
}
