import pytest
from event_core.domain.events import ChunkStored

from adapters.types import get_modal
from services.handlers import _get_namespace, _user_from_key


@pytest.mark.parametrize(
    "key,expected_user",
    (
        ("user1/test.txt", "user1"),
        ("user2/test.jpg", "user2"),
        ("user3/test.png", "user3"),
        ("user4/test.jpeg", "user4"),
        ("user5/test.mp4", "user5"),
        ("user6/a/b/c/d/test.png", "user6"),
    ),
)
def test_user_from_key(key: str, expected_user: str) -> None:
    assert _user_from_key(key) == expected_user


@pytest.mark.parametrize(
    "event,expected_namespace",
    (
        (
            ChunkStored(parent_key="user1/test.txt", key="user1/test.txt"),
            "user1__TEXT",
        ),
        (
            ChunkStored(parent_key="user2/test.mp4", key="user2/test.png"),
            "user2__VIDEO",
        ),
        (
            ChunkStored(
                parent_key="user3/test.png",
                key="user3/test.png",
            ),
            "user3__IMAGE",
        ),
    ),
)
def test_namespace_from_event(
    event: ChunkStored, expected_namespace: str
) -> None:
    user = _user_from_key(event.key)
    modal = get_modal(event.parent_key)
    namespace = _get_namespace(user, modal)
    assert namespace == expected_namespace
