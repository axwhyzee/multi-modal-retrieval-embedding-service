from pathlib import Path
from typing import Dict, List

import pytest

from adapters.repository import AbstractVectorRepo

VectorT = List[float]

TEST_DIR = Path("tests/data")


class FakeVectorRepo(AbstractVectorRepo):
    def __init__(self):
        self._namespaces: Dict[str, Dict[str, VectorT]] = {}

    def insert(self, namespace: str, key: str, vec: List[float]) -> None:
        self._namespaces[namespace] = self._namespaces.get(namespace, {})
        self._namespaces[namespace][key] = vec

    def query(
        self, namespace: str, vec: List[float], top_k: int = 5
    ) -> List[str]:
        return []


@pytest.fixture
def test_img_data() -> bytes:
    return (TEST_DIR / "test.png").read_bytes()


@pytest.fixture
def test_txt_data() -> bytes:
    return (TEST_DIR / "test.txt").read_bytes()
