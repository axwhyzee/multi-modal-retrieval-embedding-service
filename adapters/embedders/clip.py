import logging
from io import BytesIO
from typing import ClassVar, List, Optional

import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from transformers import (  # type: ignore
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from adapters.embedders.base import TextModel, VisionModel

logger = logging.getLogger(__name__)


def _norm(features: torch.Tensor):
    return features / features.norm(dim=-1, keepdim=True)


class CLIPMixin:
    EMBEDDING_DIM = 512
    _processor: ClassVar[Optional[CLIPProcessor]] = None

    def __init__(self):
        if not CLIPMixin._processor:
            logger.info("Initializing openai/clip-vit-base-patch32 processor")
            CLIPMixin._processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                local_files_only=True,
            )


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
