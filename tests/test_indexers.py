from pathlib import Path

import pytest
from event_core.domain.events import ChunkStored

from services.indexer import INDEXERS

TEST_IMG_DATA = (Path("tests/data") / "test.png").read_bytes()


@pytest.mark.parametrize(
    "event,data,expected_namespace",
    (
        (
            ChunkStored(
                parent_key="user1/test.txt", key="user1/test/1__CHUNK.txt"
            ),
            b"test content",
            "user1__TEXT",
        ),
        (
            ChunkStored(
                parent_key="user2/test.mp4", key="user2/test/1__CHUNK.png"
            ),
            TEST_IMG_DATA,
            "user2__VIDEO",
        ),
        (
            ChunkStored(
                parent_key="user3/test.png",
                key="user3/test/1__CHUNK.png",
            ),
            TEST_IMG_DATA,
            "user3__IMAGE",
        ),
    ),
)
def test_text_indexer_generates_correct_namespace_and_embedding(
    event: ChunkStored, data: bytes, expected_namespace: str
) -> None:
    suffix = Path(event.parent_key).suffix
    indexer = INDEXERS[suffix]
    vec = indexer.embed(data)
    user = event.parent_key.split("/")[0]
    namespace = indexer.get_namespace(user)

    assert type(vec) == list
    assert type(vec[0]) == float
    assert namespace == expected_namespace
