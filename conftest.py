from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import pytest

from adapters.repository import AbstractVectorRepo
from adapters.embedders import CLIPTextModel, CLIPVisionModel, DePlotModel, UniXCoderModel

VectorT = List[float]
NamespaceT = Dict[str, Dict[str, VectorT]]

TEST_DIR = Path("tests/data")


class FakeVectorRepo(AbstractVectorRepo):
    def __init__(self):
        super().__init__()
        self._namespaces: NamespaceT = defaultdict(dict)
        self._indexes: Dict[str, NamespaceT] = defaultdict(dict)

    def insert(self, index_name: str, namespace: str, key: str, vec: List[float]) -> None:
        self._indexes[index_name][namespace][key] = vec

    def query(
        self, index_name: str, namespace: str, vec: List[float], top_k: int = 5
    ) -> List[str]:
        return []


@pytest.fixture
def test_image_filepath() -> Path:
    return TEST_DIR / "image.png"


@pytest.fixture
def test_text_filepath() -> Path:
    return TEST_DIR / "text.txt"


@pytest.fixture
def test_plot_filepath() -> Path:
    return TEST_DIR / "plot.png"


@pytest.fixture
def test_code_filepath() -> Path:
    return TEST_DIR / "code.py"


@pytest.fixture(scope="session")
def text_model() -> CLIPTextModel:
    return CLIPTextModel()


@pytest.fixture(scope="session")
def vision_model() -> CLIPVisionModel:
    return CLIPVisionModel()


@pytest.fixture(scope="session")
def plot_model(text_model) -> DePlotModel:
    return DePlotModel(text_model)


@pytest.fixture(scope="session")
def code_model() -> UniXCoderModel:
    return UniXCoderModel()
