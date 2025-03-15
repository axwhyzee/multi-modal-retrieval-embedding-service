from dependency_injector import containers, providers
from event_core.adapters.services.storage import StorageAPIClient
from event_core.domain.events.elements import (
    CodeElementStored,
    ImageElementStored,
    PlotElementStored,
    TextElementStored,
    ElementStored
)
from event_core.domain.types import Element
from typing import Tuple, Type

from adapters.embedders import (
    CLIPTextModel,
    CLIPVisionModel,
    DePlotModel,
    UniXCoderModel,
)
from adapters.repository import PineconeRepo
from adapters.rerankers import BgeReranker, ColpaliReranker

MODULES = (
    "services.handlers",
    "services.factory",
)


class DIContainer(containers.DeclarativeContainer):

    # embedding models
    text_model = providers.Singleton(CLIPTextModel)
    _vision_model = providers.Singleton(CLIPVisionModel)
    _plot_model = providers.Singleton(DePlotModel, text_model)
    _code_model = providers.Singleton[UniXCoderModel]
    model_factory = providers.Dict(
        {
            CodeElementStored: _code_model,
            ImageElementStored: _vision_model,
            TextElementStored: text_model,
            PlotElementStored: _plot_model,
        }
    )

    # reranker models
    _copali_reranker = providers.Singleton(ColpaliReranker)
    _bge_reranker = providers.Singleton(BgeReranker)
    reranker_factory = providers.Dict(
        {
            # Element.IMAGE: _copali_reranker,
            Element.TEXT: _bge_reranker,
        }
    )

    # external services
    vec_repo = providers.Singleton(
        PineconeRepo,
        (
            Element.CODE.value,
            Element.IMAGE.value,
            Element.PLOT.value,
            Element.TEXT.value,
        ),
        (
            UniXCoderModel.EMBEDDING_DIM,
            CLIPVisionModel.EMBEDDING_DIM,
            DePlotModel.EMBEDDING_DIM,
            CLIPTextModel.EMBEDDING_DIM,
        ),
    )
    storage = providers.Singleton(StorageAPIClient)


def bootstrap(lazy_load: bool = True) -> None:
    container = DIContainer()

    if not lazy_load:
        # avoid lazy instantiation which times out requests
        container.reranker_factory()
        container.model_factory()
    container.wire(modules=MODULES)
