from pathlib import Path
from typing import List

from event_core.adapters.storage import get
from event_core.domain.events import ChunkStored

from adapters.model import CLIP
from adapters.repository import PineconeRepo

repo = PineconeRepo()
model = CLIP

IMAGE_FILE_EXTS = (".jpg", ".jpeg", ".png")
TEXT_FILE_EXTS = ".txt"


def handle_chunk(event: ChunkStored) -> None:
    obj = get(event.obj_path)
    suffix = Path(event.obj_path).suffix

    # map obj to corresponding embedding function
    if suffix in IMAGE_FILE_EXTS:
        emb = model.embed_image(obj)
    elif suffix in TEXT_FILE_EXTS:
        emb = model.embed_text(obj.decode("utf-8"))
    else:
        # TODO: emit error event
        pass

    # store in vector db
    repo.insert(event.obj_path, emb)


def handle_text_query(user: str, text: str) -> List[str]:
    vec =  model.embed_text(text)
    return repo.query(user, vec)
