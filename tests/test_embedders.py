import pytest
from event_core.domain.events import ChunkStored
from event_core.domain.types import Modal

from config import EMBEDDING_DIM
from services.factory import ModalToChunkEmbedder


@pytest.mark.parametrize(
    "event,fixture_data",
    (
        (
            ChunkStored(key="user1/test.txt", modal=Modal.TEXT),
            "test_txt_data",
        ),
        (
            ChunkStored(key="user2/test.png", modal=Modal.IMAGE),
            "test_img_data",
        ),
        (
            ChunkStored(key="user3/test.png", modal=Modal.IMAGE),
            "test_img_data",
        ),
    ),
)
def test_embedder_generates_vector_with_correct_dim(
    event: ChunkStored, fixture_data: str, request: pytest.FixtureRequest
) -> None:
    data = request.getfixturevalue(fixture_data)
    modal = event.modal
    embedder = ModalToChunkEmbedder[modal]
    vec = embedder.embed(data)

    assert type(vec) == list
    assert type(vec[0]) == float
    assert len(vec) == EMBEDDING_DIM
