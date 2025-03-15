import logging
from io import BytesIO
from typing import Iterable, Iterator

import torch
from colpali_engine.models import ColPali, ColPaliProcessor  # type: ignore
from PIL import Image

from adapters.rerankers.base import TOP_K, AbstractReranker
from config import DEVICE

logger = logging.getLogger(__name__)


class ColpaliReranker(AbstractReranker):
    def __init__(self):
        logger.info("Initializing vidore/colpali-v1.2")
        self._model = ColPali.from_pretrained(
            "vidore/colpali-v1.2",
            torch_dtype=torch.bfloat16,
            device_map=DEVICE,
            local_files_only=True,
        ).eval()
        self._processor = ColPaliProcessor.from_pretrained(
            "vidore/colpali-v1.2",
            local_files_only=True,
        )

    def rerank(
        self, query: str, candidates: Iterator[bytes], top_k: int = TOP_K
    ) -> Iterable[int]:
        BATCH_SIZE = 8

        scores = []
        text_input = self._processor.process_queries([query]).to(DEVICE)
        with torch.no_grad():
            text_emb = self._model(**text_input)

        done = False
        while not done:
            batch_imgs = []

            for _ in range(BATCH_SIZE):
                try:
                    img = Image.open(BytesIO(next(candidates)))
                    batch_imgs.append(img)
                except StopIteration:
                    done = True
                    break

            if not batch_imgs:
                break

            batch_img_inputs = self._processor.process_images(batch_imgs).to(
                DEVICE
            )
            with torch.no_grad():
                img_embs = self._model(**batch_img_inputs)

            batch_scores = self._processor.score_multi_vector(
                text_emb, img_embs
            ).reshape(-1)
            scores.extend(batch_scores)
        logger.info(scores)
        idx_scores = zip(scores, range(len(scores)))
        return [i for _, i in sorted(idx_scores, reverse=True)[:top_k]]
