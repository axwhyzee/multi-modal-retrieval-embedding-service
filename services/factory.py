from dependency_injector.wiring import Provide, inject
from event_core.domain.events import (
    ElementStored,
    ImageElementStored,
    PlotElementStored,
    TextElementStored,
)

from adapters.embedder import AbstractEmbeddingModel
from bootstrap import DIContainer


class UnavailableModel(Exception): ...


@inject
def model_factory(
    event: ElementStored,
    vision_model=Provide[DIContainer.vision_model],
    text_model=Provide[DIContainer.text_model],
    plot_model=Provide[DIContainer.plot_model],
) -> AbstractEmbeddingModel:
    if isinstance(event, ImageElementStored):
        return vision_model
    elif isinstance(event, TextElementStored):
        return text_model
    elif isinstance(event, PlotElementStored):
        return plot_model
    raise UnavailableModel(f"No available model for event {event}")
