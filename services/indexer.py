from abc import abstractmethod
from typing import Dict, List, Type, Protocol, ClassVar

from adapters.embedder import CLIP
from adapters.types import Modality

model = CLIP


class AbstractIndexerFactory(Protocol):
    modality: ClassVar[Modality]

    @classmethod
    @abstractmethod
    def embed(cls, obj: bytes) -> List[float]:
        raise NotImplementedError

    @classmethod
    def get_namespace(cls, user: str) -> str:
        return f"{user}__{cls.modality}"


class TextIndexer(AbstractIndexerFactory):
    modality = Modality.TEXT

    @classmethod
    def embed(cls, obj: bytes) -> List[float]:
        return model.embed_text(obj.decode("utf-8"))


class ImageIndexer(AbstractIndexerFactory):
    modality = Modality.IMAGE

    @classmethod
    def embed(cls, obj: bytes) -> List[float]:
        return model.embed_image(obj)


class VideoIndexer(AbstractIndexerFactory):
    modality = Modality.VIDEO

    @classmethod
    def embed(cls, obj: bytes) -> List[float]:
        return model.embed_image(obj)


INDEXERS: Dict[str, Type[AbstractIndexerFactory]] = {
    ".jpg": ImageIndexer,
    ".jpeg": ImageIndexer,
    ".png": ImageIndexer,
    ".txt": TextIndexer,
    ".mp4": VideoIndexer,
}
