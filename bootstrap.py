from dependency_injector import containers, providers
from event_core.adapters.services.storage import StorageAPIClient
from event_core.domain.types import Modal

from adapters.embedder import CLIPTextModel, CLIPVisionModel, DePlotModel
from adapters.repository import PineconeRepo
from adapters.reranker import BgeReranker, ColpaliReranker

MODULES = (
    "services.handlers",
    "services.factory",
)


class DIContainer(containers.DeclarativeContainer):
    vec_repo = providers.Singleton(PineconeRepo)
    text_model = providers.Singleton(CLIPTextModel)
    vision_model = providers.Singleton(CLIPVisionModel)
    plot_model = providers.Singleton(DePlotModel, text_model)
    storage = providers.Singleton(StorageAPIClient)

    _copali_reranker = providers.Singleton(ColpaliReranker)
    _bge_reranker = providers.Singleton(BgeReranker)
    rerankers = providers.Dict(
        {
            Modal.IMAGE: _copali_reranker,
            Modal.TEXT: _bge_reranker,
        }
    )


def bootstrap(lazy_load: bool = True) -> None:
    container = DIContainer()

    if not lazy_load:
        # avoid lazy instantiation which times out requests
        container.text_model()
        container.vision_model()
        container.plot_model()
        container.rerankers()
    container.wire(modules=MODULES)
