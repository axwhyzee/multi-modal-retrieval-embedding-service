from typing import Dict, List

from adapters.repository import AbstractVectorRepo

VectorT = List[float]


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
