from pathlib import Path
from typing import List

from event_core.adapters.services.api.storage import get
from event_core.domain.events import ChunkStored

from adapters.repository import PineconeRepo
from services.indexer import INDEXERS, model

repo = PineconeRepo()


def handle_chunk(event: ChunkStored) -> None:
    key = event.key
    user = key.split("/")[0]
    data = get(key)
    suffix = Path(key).suffix

    indexer = INDEXERS[suffix]
    vec = indexer.embed(data)
    namespace = indexer.get_namespace(user)
    repo.insert(namespace, key, vec)


def handle_text_query(user: str, text: str) -> List[str]:
    vec = model.embed_text(text)
    return repo.query(user, vec)


def handle_embed_text(text: str) -> List[float]:
    return model.embed_text(text)
