import pytest
from event_core.domain.events import (
    ElementStored,
    ImageElementStored,
    PlotElementStored,
    TextElementStored,
)
from event_core.domain.types import path_to_ext

from bootstrap import bootstrap
from config import EMBEDDING_DIM
from services.factory import model_factory


@pytest.mark.parametrize(
    "event,fixture_data",
    (
        (
            TextElementStored(key="user1/text.txt"),
            "test_txt_data",
        ),
        (
            ImageElementStored(key="user2/img.png"),
            "test_img_data",
        ),
        (PlotElementStored(key="user3/table.png"), "test_table_data"),
    ),
)
def test_embedder_generates_vector_with_correct_dim(
    event: ElementStored, fixture_data: str, request: pytest.FixtureRequest
) -> None:
    bootstrap()
    data = request.getfixturevalue(fixture_data)
    model = model_factory(event)
    vec = model.embed(data)

    assert type(vec) == list
    assert type(vec[0]) == float
    assert len(vec) == EMBEDDING_DIM
