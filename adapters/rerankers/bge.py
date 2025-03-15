import logging
from typing import Iterable, Iterator

from transformers import (  # type: ignore
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from adapters.rerankers.base import TOP_K, AbstractReranker

logger = logging.getLogger(__name__)


class BgeReranker(AbstractReranker):
    def __init__(self):
        logger.info("Initializing BAAI/bge-reranker-v2-m3")
        self._tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/bge-reranker-v2-m3",
            local_files_only=True,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            "BAAI/bge-reranker-v2-m3",
            local_files_only=True,
        ).eval()

    def rerank(
        self, query: str, candidates: Iterator[bytes], top_k: int = TOP_K
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
