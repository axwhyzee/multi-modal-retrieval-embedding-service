import logging
from abc import ABC, abstractmethod
from io import BytesIO
from typing import List, Optional

import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from transformers import (  # type: ignore
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    Pix2StructForConditionalGeneration,
    Pix2StructProcessor,
)

logger = logging.getLogger(__name__)


def _norm(features: torch.Tensor):
    return features / features.norm(dim=-1, keepdim=True)


class AbstractEmbeddingModel(ABC):

    @abstractmethod
    def embed(self, data: bytes) -> List[float]:
        raise NotImplementedError


class TextModel(AbstractEmbeddingModel): ...


class VisionModel(AbstractEmbeddingModel): ...


class PlotModel(AbstractEmbeddingModel): ...


class CLIPMixin:
    """Singleton CLIP model for shared processor"""

    _instance: Optional["CLIPMixin"] = None
    _processor: CLIPProcessor

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            logger.info("Initializing openai/clip-vit-base-patch32 processor")
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                local_files_only=True,
            )
        return cls._instance


class CLIPVisionModel(VisionModel, CLIPMixin):
    def __init__(self):
        logger.info("Initializing openai/clip-vit-base-patch32 vision model")
        super().__init__()
        self._model = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-base-patch32",
            local_files_only=True,
        )

    def embed(self, data: bytes) -> List[float]:
        image = Image.open(BytesIO(data))
        inputs = self._processor(
            images=image, return_tensors="pt", padding=True
        )
        outputs = self._model(**inputs)
        emb: torch.Tensor
        emb = outputs.image_embeds
        emb = emb.reshape(-1)
        emb = _norm(emb)
        return emb.tolist()


class CLIPTextModel(TextModel, CLIPMixin):
    def __init__(self):
        logger.info("Initializing openai/clip-vit-base-patch32 text model")
        super().__init__()
        self._model = CLIPTextModelWithProjection.from_pretrained(
            "openai/clip-vit-base-patch32",
            local_files_only=True,
        )

    def embed(self, data: bytes) -> List[float]:
        text = data.decode("utf-8")
        text_splitter = (
            RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                self._processor.tokenizer,
                chunk_size=77,  # CLIP has hard limit of 77
                chunk_overlap=15,
            )
        )
        texts = text_splitter.split_text(text)
        inputs = self._processor(text=texts, return_tensors="pt", padding=True)
        outputs = self._model(**inputs)
        emb: torch.Tensor
        emb = outputs.text_embeds
        emb = emb.mean(dim=0)
        emb = emb.reshape(-1)
        emb = _norm(emb)
        return emb.tolist()


class DePlotModel(PlotModel):

    def __init__(self, text_model: TextModel):
        logger.info("Initializing google/deplot")
        self._processor = Pix2StructProcessor.from_pretrained("google/deplot")
        self._deplot_model = (
            Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
        )
        self._text_model = text_model

    def embed(self, data: bytes) -> List[float]:
        image = Image.open(BytesIO(data))
        inputs = self._processor(
            images=image,
            text="Generate underlying data table of the figure below:",
            return_tensors="pt",
        )
        predictions = self._deplot_model.generate(**inputs, max_new_tokens=512)
        table_str = self._processor.decode(
            predictions[0], skip_special_tokens=True
        )
        return self._text_model.embed(table_str.encode("utf-8"))
