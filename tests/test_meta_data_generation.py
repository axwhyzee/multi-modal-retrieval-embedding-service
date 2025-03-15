import pytest
from event_core.domain.events.elements import (
    ElementStored,
    ImageElementStored,
    PlotElementStored,
    TextElementStored,
    CodeElementStored,
)
from event_core.domain.events.elements import ELEM_TYPES

from handlers import _get_vec_repo_namespace, _user_from_key


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
            TextElementStored(key="user1/test.txt"),
            "user1__TEXT",
        ),
        (
            ImageElementStored(key="user2/test.png"),
            "user2__IMAGE",
        ),
        (
            PlotElementStored(key="user3/table.png"),
            "user3__PLOT",
        ),
        (
            CodeElementStored(key="user4/code.py"),
            "user4__CODE",
        ),
    ),
)
def test_namespace_from_event(
    event: ElementStored, expected_namespace: str
) -> None:
    user = _user_from_key(event.key)
    elem = ELEM_TYPES[event.__class__]
    namespace = _get_vec_repo_namespace(user, elem)
    assert namespace == expected_namespace
