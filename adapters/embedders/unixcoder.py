from typing import List

import torch

from adapters.embedders._unixcoder import UniXcoder
from adapters.embedders.base import CodeModel
from config import DEVICE


class UniXCoderModel(CodeModel):

    EMBEDDING_DIM = 768

    def __init__(self):
        self._model = UniXcoder("microsoft/unixcoder-base")
        self._model.to(DEVICE)

    def embed(self, data: bytes) -> List[float]:
        tokens_ids = self._model.tokenize(
            [data.decode("utf-8")], max_length=512, mode="<encoder-only>"
        )
        source_ids = torch.tensor(tokens_ids).to(DEVICE)
        with torch.no_grad():
            _, embedding = self._model(source_ids)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.reshape(-1).tolist()
