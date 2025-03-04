import pytest
from event_core.domain.events import ChunkStored
from event_core.domain.types import PRIMITIVE_EXT_TO_MODAL, Modal, path_to_ext

from bootstrap import bootstrap
from config import EMBEDDING_DIM
from services.factory import ModalToChunkEmbedder


@pytest.mark.parametrize(
    "event,fixture_data",
    (
        (
            ChunkStored(key="user1/test.txt"),
            "test_txt_data",
        ),
        (
            ChunkStored(key="user2/test.png"),
            "test_img_data",
        ),
        (
            ChunkStored(key="user3/test.png"),
            "test_img_data",
        ),
    ),
)
def test_embedder_generates_vector_with_correct_dim(
    event: ChunkStored, fixture_data: str, request: pytest.FixtureRequest
) -> None:
    bootstrap()
    data = request.getfixturevalue(fixture_data)
    ext = path_to_ext(event.key)
    modal = PRIMITIVE_EXT_TO_MODAL[ext]
    embedder = ModalToChunkEmbedder[modal]
    vec = embedder.embed(data)

    assert type(vec) == list
    assert type(vec[0]) == float
    assert len(vec) == EMBEDDING_DIM
