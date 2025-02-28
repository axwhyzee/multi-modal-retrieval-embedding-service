import logging
from typing import Dict, List, TypeAlias

from dependency_injector.wiring import Provide, inject
from event_core.adapters.services.api.storage import get
from event_core.domain.events import ChunkStored
from event_core.domain.types import Modal

from adapters.embedder import AbstractEmbeddingModel
from adapters.repository import AbstractVectorRepo
from adapters.reranker import AbstractDualModalReranker
from bootstrap import DIContainer
from services.factory import ModalToChunkEmbedder

logger = logging.getLogger(__name__)

KeysT: TypeAlias = List[str]


def _user_from_key(key: str) -> str:
    return key.split("/")[0]


def _get_namespace(user: str, modal: Modal) -> str:
    return f"{user}__{modal}"


@inject
def handle_chunk(
    event: ChunkStored,
    vec_repo: AbstractVectorRepo = Provide[DIContainer.vec_repo],
) -> None:
    logger.info(f"Handling chunk {event=}")
    key = event.key
    modal = event.modal
    user = _user_from_key(key)

    embedder = ModalToChunkEmbedder[modal]
    vec = embedder.embed(get(key))
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
    rerankers: Dict[Modal, AbstractDualModalReranker] = Provide[
        DIContainer.rerankers
    ],
) -> Dict[Modal, KeysT]:
    logger.info(f"Handling query text {user=} {text=}")
    res: Dict[Modal, List[str]] = {}
    vec = emb_model.embed_text(text)
    for modal in Modal:
        namespace = _get_namespace(user, modal)
        keys = vec_repo.query(namespace, vec, n_cands)
        logger.info(f"Candidates: {keys}")
        if not keys:
            res[modal] = []
            continue
        reranker = rerankers[modal]
        ranks = reranker.rerank(text, [get(key) for key in keys], n_rank)
        res[modal] = [keys[i] for i in ranks]
    return res
