import logging
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Iterable

import torch
from colpali_engine.models import ColPali, ColPaliProcessor  # type: ignore
from PIL import Image
from transformers import (  # type: ignore
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from config import CACHE_DIR

logger = logging.getLogger(__name__)

TOP_K = 5


class AbstractReranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, candidates: Iterable[bytes], top_k: int = TOP_K
    ) -> Iterable[int]:
        raise NotImplementedError


class BgeReranker(AbstractReranker):
    def __init__(self):
        logger.info("Initializing BAAI/bge-reranker-v2-m3")
        self._tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/bge-reranker-v2-m3",
            cache_dir=CACHE_DIR,
            local_files_only=True,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            "BAAI/bge-reranker-v2-m3",
            cache_dir=CACHE_DIR,
            local_files_only=True,
        ).eval()

    def rerank(
        self, query: str, candidates: Iterable[bytes], top_k: int = TOP_K
    ) -> Iterable[int]:
        pairs = [
            (query, candidate.decode("utf-8")) for candidate in candidates
        ]
        inputs = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        scores = (
            self._model(**inputs, return_dict=True)
            .logits.view(
                -1,
            )
            .float()
        )
        idx_scores = zip(scores, range(len(scores)))
        return [i for _, i in sorted(idx_scores, reverse=True)[:top_k]]


class ColpaliReranker(AbstractReranker):
    def __init__(self):
        logger.info("Initializing vidore/colpali-v1.2")
        self._model = ColPali.from_pretrained(
            "vidore/colpali-v1.2", 
            torch_dtype=torch.bfloat16, 
            device_map="mps",
            cache_dir=CACHE_DIR,
            local_files_only=True,
        ).eval()
        self._processor = ColPaliProcessor.from_pretrained(
            "vidore/colpali-v1.2",
            cache_dir=CACHE_DIR,
            local_files_only=True,
        )

    def rerank(
        self, query: str, candidates: Iterable[bytes], top_k: int = TOP_K
    ) -> Iterable[int]:
        images = [Image.open(BytesIO(cand)) for cand in candidates]
        batch_images = self._processor.process_images(images).to(
            self._model.device
        )
        batch_queries = self._processor.process_queries([query]).to(
            self._model.device
        )

        with torch.no_grad():
            image_embeddings = self._model(**batch_images)
            query_embeddings = self._model(**batch_queries)

        scores = self._processor.score_multi_vector(
            query_embeddings, image_embeddings
        ).reshape(-1)
        idx_scores = zip(scores, range(len(scores)))
        return [i for _, i in sorted(idx_scores, reverse=True)[:top_k]]
