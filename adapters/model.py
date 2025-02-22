from abc import ABC, abstractmethod
from io import BytesIO
from typing import List

import torch
from PIL import Image
from transformers import (  # type: ignore
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from config import MODEL_PATH


def _norm(features: torch.Tensor):
    return features / features.norm(dim=-1, keepdim=True)


class AbstractModel(ABC):
    @classmethod
    @abstractmethod
    def embed_text(cls, text: str) -> List[float]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def embed_image(cls, img: bytes) -> List[float]:
        raise NotImplementedError


class CLIP(AbstractModel):
    _text_model = CLIPTextModelWithProjection.from_pretrained(MODEL_PATH)
    _vision_model = CLIPVisionModelWithProjection.from_pretrained(MODEL_PATH)
    _processor = CLIPProcessor.from_pretrained(MODEL_PATH)

    @classmethod
    def embed_text(cls, text: str) -> List[float]:
        inputs = cls._processor(text=[text], return_tensors="pt", padding=True)
        outputs = cls._text_model(**inputs)
        emb: torch.Tensor
        emb = outputs.text_embeds
        emb = emb.reshape(-1)
        emb = _norm(emb)
        return emb.tolist()

    @classmethod
    def embed_image(cls, img: bytes) -> List[float]:
        image = Image.open(BytesIO(img))
        inputs = cls._processor(
            images=image, return_tensors="pt", padding=True
        )
        outputs = cls._vision_model(**inputs)
        emb: torch.Tensor
        emb = outputs.image_embeds
        emb = emb.reshape(-1)
        emb = _norm(emb)
        return emb.tolist()
