from typing import Dict, List

from dependency_injector import containers, providers

from adapters.embedder import CLIPEmbedder
from adapters.repository import PineconeRepo
from adapters.reranker import (
    AbstractDualModalReranker,
    BgeReranker,
    ColpaliReranker,
)
from adapters.types import Modal


class DIContainer(containers.DeclarativeContainer):
    vec_repo = providers.Singleton(PineconeRepo)
    emb_model = providers.Singleton(CLIPEmbedder)

    _copali_reranker = providers.Singleton(ColpaliReranker)
    _bge_reranker = providers.Singleton(BgeReranker)
    rerankers = providers.Dict(
        {
            Modal.IMAGE: _copali_reranker,
            Modal.VIDEO: _copali_reranker,
            Modal.TEXT: _bge_reranker,
        }
    )


def bootstrap() -> None:
    container = DIContainer()
    container.wire(modules=["services.handlers", "services.factory"])
