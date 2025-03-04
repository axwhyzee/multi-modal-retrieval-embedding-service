from dependency_injector import containers, providers
from event_core.adapters.services.storage import StorageAPIClient
from event_core.domain.types import Modal

from adapters.embedder import CLIPEmbedder
from adapters.repository import PineconeRepo
from adapters.reranker import BgeReranker, ColpaliReranker, FakeReranker

MODULES = (
    "services.handlers",
    "services.factory",
)


class DIContainer(containers.DeclarativeContainer):
    vec_repo = providers.Singleton(PineconeRepo)
    emb_model = providers.Singleton(CLIPEmbedder)
    storage = providers.Singleton(StorageAPIClient)

    _copali_reranker = providers.Singleton(ColpaliReranker)
    _bge_reranker = providers.Singleton(BgeReranker)
    rerankers = providers.Dict(
        {
            Modal.IMAGE: _copali_reranker,
            Modal.TEXT: _bge_reranker,
        }
    )


def bootstrap() -> None:
    container = DIContainer()
    # avoid lazy instantiation which times out requests
    container.emb_model()
    container.rerankers()
    container.wire(modules=MODULES)
