import pytest
from event_core.domain.events import ChunkStored
from event_core.domain.types import Modal

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
            ChunkStored(key="user1/test.txt", modal=Modal.TEXT),
            "user1__TEXT",
        ),
        (
            ChunkStored(key="user2/test.png", modal=Modal.VIDEO),
            "user2__VIDEO",
        ),
        (
            ChunkStored(key="user3/test.png", modal=Modal.IMAGE),
            "user3__IMAGE",
        ),
    ),
)
def test_namespace_from_event(
    event: ChunkStored, expected_namespace: str
) -> None:
    user = _user_from_key(event.key)
    modal = event.modal
    namespace = _get_namespace(user, modal)
    assert namespace == expected_namespace
