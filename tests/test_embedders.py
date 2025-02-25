import pytest
from event_core.domain.events import ChunkStored

from adapters.types import get_modal
from config import EMBEDDING_DIM
from services.factory import ModalToChunkEmbedder


@pytest.mark.parametrize(
    "event,fixture_data",
    (
        (
            ChunkStored(parent_key="user1/test.txt", key="user1/test.txt"),
            "test_txt_data",
        ),
        (
            ChunkStored(parent_key="user2/test.mp4", key="user2/test.png"),
            "test_img_data",
        ),
        (
            ChunkStored(
                parent_key="user3/test.png",
                key="user3/test.png",
            ),
            "test_img_data",
        ),
    ),
)
def test_embedder_generates_vector_with_correct_dim(
    event: ChunkStored, fixture_data: str, request: pytest.FixtureRequest
) -> None:
    data = request.getfixturevalue(fixture_data)
    modal = get_modal(event.parent_key)
    embedder = ModalToChunkEmbedder[modal]
    vec = embedder.embed(data)

    assert type(vec) == list
    assert type(vec[0]) == float
    assert len(vec) == EMBEDDING_DIM
