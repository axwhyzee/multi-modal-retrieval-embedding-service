import logging
from abc import ABC, abstractmethod
from io import BytesIO
from typing import List

import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
from transformers import (  # type: ignore
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

logger = logging.getLogger(__name__)


def _norm(features: torch.Tensor):
    return features / features.norm(dim=-1, keepdim=True)


class AbstractEmbeddingModel(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_image(self, img: bytes) -> List[float]:
        raise NotImplementedError


class CLIPEmbedder(AbstractEmbeddingModel):
    def __init__(self):
        logger.info("Initializing openai/clip-vit-base-patch32")
        self._text_model = CLIPTextModelWithProjection.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self._vision_model = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self._processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def embed_text(self, text: str) -> List[float]:
        text_splitter = (
            RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                self._processor.tokenizer,
                chunk_size=77,  # CLIP has hard limit of 77
                chunk_overlap=15,
            )
        )
        texts = text_splitter.split_text(text)
        inputs = self._processor(text=texts, return_tensors="pt", padding=True)
        outputs = self._text_model(**inputs)
        emb: torch.Tensor
        emb = outputs.text_embeds
        emb = emb.mean(dim=0)
        emb = emb.reshape(-1)
        emb = _norm(emb)
        return emb.tolist()

    def embed_image(self, img: bytes) -> List[float]:
        image = Image.open(BytesIO(img))
        inputs = self._processor(
            images=image, return_tensors="pt", padding=True
        )
        outputs = self._vision_model(**inputs)
        emb: torch.Tensor
        emb = outputs.image_embeds
        emb = emb.reshape(-1)
        emb = _norm(emb)
        return emb.tolist()
