import logging
from typing import Dict, Iterable, Iterator, List, TypeAlias

from dependency_injector.wiring import Provide, inject
from event_core.adapters.services.storage import StorageClient
from event_core.domain.events import ChunkStored
from event_core.domain.types import EXT_TO_MODAL, Modal, path_to_ext

from adapters.embedder import AbstractEmbeddingModel
from adapters.repository import AbstractVectorRepo
from adapters.reranker import AbstractReranker
from bootstrap import DIContainer
from config import TOP_N_MULTIPLIER
from services.factory import ModalToChunkEmbedder

logger = logging.getLogger(__name__)

KeysT: TypeAlias = List[str]


def _user_from_key(key: str) -> str:
    return key.split("/")[0]


def _get_vec_repo_namespace(user: str, modal: Modal) -> str:
    return f"{user}__{modal}"


@inject
def handle_chunk(
    event: ChunkStored,
    vec_repo: AbstractVectorRepo = Provide[DIContainer.vec_repo],
    storage: StorageClient = Provide[DIContainer.storage],
) -> None:
    logger.info(f"Handling chunk {event=}")
    user = _user_from_key(event.key)
    ext = path_to_ext(event.key)
    modal = EXT_TO_MODAL[ext]
    embedder = ModalToChunkEmbedder[modal]
    vec = embedder.embed(storage[event.key])
    namespace = _get_vec_repo_namespace(user, modal)
    vec_repo.insert(namespace, event.key, vec)


@inject
def handle_query_text(
    user: str,
    text: str,
    top_n: int,
    vec_repo: AbstractVectorRepo = Provide[DIContainer.vec_repo],
    emb_model: AbstractEmbeddingModel = Provide[DIContainer.emb_model],
    rerankers: Dict[Modal, AbstractReranker] = Provide[DIContainer.rerankers],
    storage: StorageClient = Provide[DIContainer.storage],
) -> KeysT:

    def _generate_objs(keys: Iterable[str]) -> Iterator[bytes]:
        for key in keys:
            yield storage[key]

    logger.info(f"Handling query text {user=} {text=}")
    res: KeysT = []
    vec = emb_model.embed_text(text)

    for modal in Modal:
        # query vector repo for candidates
        namespace = _get_vec_repo_namespace(user, modal)
        keys = vec_repo.query(namespace, vec, top_n * TOP_N_MULTIPLIER)
        logger.info(f"Found {len(keys)} candidates")
        if len(keys) < top_n:
            res.extend(keys)
            continue

        # rerank candidates
        reranker = rerankers[modal]
        ranks = reranker.rerank(text, _generate_objs(keys), top_n)
        res.extend([keys[i] for i in ranks])
    return res
