from pathlib import Path
from typing import Dict, List, TypeAlias

from event_core.adapters.services.api.storage import get
from event_core.domain.events import ChunkStored

from adapters.repository import PineconeRepo
from adapters.reranker import RERANKERS
from adapters.types import Modality
from services.indexer import INDEXERS, model

repo = PineconeRepo()

KeysT: TypeAlias = List[str]


def _get_user_from_key(key: str) -> str:
    return key.split("/")[0]


def handle_chunk(event: ChunkStored) -> None:
    key = event.key
    user = _get_user_from_key(key)
    data = get(key)
    suffix = Path(event.parent_key).suffix

    indexer = INDEXERS[suffix]
    vec = indexer.embed(data)
    namespace = indexer.get_namespace(user)
    repo.insert(namespace, key, vec)


def handle_query_text(user: str, text: str) -> Dict[Modality, KeysT]:
    res = {}
    vec = model.embed_text(text)

    for indexer in INDEXERS.values():
        modality = indexer.modality
        namespace = indexer.get_namespace(user)
        keys = repo.query(namespace, vec)

        reranker = RERANKERS[modality]
        ranks = reranker.rerank(text, [get(key) for key in keys])
        res[modality] = [keys[i] for i in ranks]

    return res
