from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar

from transformers import (  # type: ignore
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from config import TEXT_TEXT_RERANKER_MODEL

T = TypeVar("T")


class AbstractDualModalReranker(ABC, Generic[T]):
    @abstractmethod
    def rerank(
        self, query: str, candidates: Iterable[T], top_k: int = 5
    ) -> Iterable[int]:
        raise NotImplementedError


class TextTextReranker(AbstractDualModalReranker[str]): ...


class BgeReranker(TextTextReranker):
    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained(
            TEXT_TEXT_RERANKER_MODEL
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            TEXT_TEXT_RERANKER_MODEL
        )
        self._model.eval()

    def rerank(
        self, query: str, candidates: Iterable[str], top_k: int = 5
    ) -> Iterable[int]:
        pairs = [(query, candidate) for candidate in candidates]
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
        idx_scores = zip(range(len(scores)), scores)
        return [i for i, _ in sorted(idx_scores, reverse=True)[:top_k]]
