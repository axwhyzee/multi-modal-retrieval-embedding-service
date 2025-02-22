from abc import ABC, abstractmethod
from typing import Dict, List, Type

from adapters.embedder import CLIP
from adapters.types import Modality

model = CLIP


class AbstractIndexerFactory(ABC):
    @classmethod
    @abstractmethod
    def embed(cls, obj: bytes) -> List[float]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_modality(cls) -> Modality:
        raise NotImplementedError

    @classmethod
    def get_namespace(cls, user: str) -> str:
        return f"{user}__{cls.get_modality()}"


class TextIndexer(AbstractIndexerFactory):
    @classmethod
    def embed(cls, obj: bytes) -> List[float]:
        return model.embed_text(obj.decode("utf-8"))

    @classmethod
    def get_modality(cls) -> Modality:
        return Modality.TEXT


class ImageIndexer(AbstractIndexerFactory):
    @classmethod
    def embed(cls, obj: bytes) -> List[float]:
        return model.embed_image(obj)

    @classmethod
    def get_modality(cls) -> Modality:
        return Modality.IMAGE


class VideoIndexer(AbstractIndexerFactory):
    @classmethod
    def embed(cls, obj: bytes) -> List[float]:
        return model.embed_image(obj)

    @classmethod
    def get_modality(cls) -> Modality:
        return Modality.VIDEO


INDEXERS: Dict[str, Type[AbstractIndexerFactory]] = {
    ".jpg": ImageIndexer,
    ".jpeg": ImageIndexer,
    ".png": ImageIndexer,
    ".txt": TextIndexer,
    ".mp4": VideoIndexer,
}
