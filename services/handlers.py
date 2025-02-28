import logging
from typing import Dict, Iterable, Iterator, List, TypeAlias

from dependency_injector.wiring import Provide, inject
from event_core.adapters.services.storage import StorageClient
from event_core.domain.events import ChunkStored
from event_core.domain.types import Modal

from adapters.embedder import AbstractEmbeddingModel
from adapters.repository import AbstractVectorRepo
from adapters.reranker import AbstractReranker
from bootstrap import DIContainer
from services.factory import ModalToChunkEmbedder

logger = logging.getLogger(__name__)

KeysT: TypeAlias = List[str]


def _user_from_key(key: str) -> str:
    return key.split("/")[0]


def _get_namespace(user: str, modal: Modal) -> str:
    return f"{user}__{modal}"


def _obj_generator(
    storage_client: StorageClient, keys: Iterable[str]
) -> Iterator[bytes]:
    for key in keys:
        yield storage_client.get(key)


@inject
def handle_chunk(
    event: ChunkStored,
    vec_repo: AbstractVectorRepo = Provide[DIContainer.vec_repo],
    storage_client: StorageClient = Provide[DIContainer.storage_client],
) -> None:
    logger.info(f"Handling chunk {event=}")
    key = event.key
    modal = event.modal
    user = _user_from_key(key)

    embedder = ModalToChunkEmbedder[modal]
    vec = embedder.embed(storage_client.get(key))
    namespace = _get_namespace(user, modal)
    vec_repo.insert(namespace, key, vec)


@inject
def handle_query_text(
    user: str,
    text: str,
    n_cands: int,
    n_rank: int,
    vec_repo: AbstractVectorRepo = Provide[DIContainer.vec_repo],
    emb_model: AbstractEmbeddingModel = Provide[DIContainer.emb_model],
    rerankers: Dict[Modal, AbstractReranker] = Provide[DIContainer.rerankers],
    storage_client: StorageClient = Provide[DIContainer.storage_client],
) -> Dict[Modal, KeysT]:
    logger.info(f"Handling query text {user=} {text=}")
    res: Dict[Modal, List[str]] = {}
    vec = emb_model.embed_text(text)

    for modal in Modal:
        # query vector repo for candidates
        namespace = _get_namespace(user, modal)
        keys = vec_repo.query(namespace, vec, n_cands)
        logger.info(f"Candidates: {keys}")
        if not keys:
            res[modal] = []
            continue

        # rerank the candidates
        reranker = rerankers[modal]
        ranks = reranker.rerank(
            text, _obj_generator(storage_client, keys), n_rank
        )
        res[modal] = [keys[i] for i in ranks]
    return res
