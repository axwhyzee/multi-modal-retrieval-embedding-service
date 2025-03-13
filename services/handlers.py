import logging
from typing import Dict, Iterable, Iterator, List, TypeAlias

from dependency_injector.wiring import Provide, inject
from event_core.adapters.services.storage import StorageClient
from event_core.domain.events import ElementStored
from event_core.domain.types import EXT_TO_MODAL, Modal, path_to_ext

from adapters.embedder import AbstractEmbeddingModel
from adapters.repository import AbstractVectorRepo
from adapters.reranker import AbstractReranker
from bootstrap import DIContainer
from config import TOP_N_MULTIPLIER
from services.factory import model_factory

logger = logging.getLogger(__name__)


KeysT: TypeAlias = List[str]


def _user_from_key(key: str) -> str:
    return key.split("/")[0]


def _get_vec_repo_namespace(user: str, modal: Modal) -> str:
    return f"{user}__{modal}"


@inject
def handle_element(
    event: ElementStored,
    vec_repo: AbstractVectorRepo = Provide[DIContainer.vec_repo],
    storage: StorageClient = Provide[DIContainer.storage],
) -> None:
    """
    1. Fetch element object
    2. Create corresponding embedding model
    3. Embed element
    4. Insert into vec repo at the corresponding namespace
    """
    logger.info(f"Handling ElementStored: {event=}")
    key = event.key
    user = _user_from_key(key)
    ext = path_to_ext(key)
    modal = EXT_TO_MODAL[ext]
    model = model_factory(event)
    vec = model.embed(storage[key])
    namespace = _get_vec_repo_namespace(user, modal)
    vec_repo.insert(namespace, key, vec)


@inject
def handle_query_text(
    user: str,
    text: str,
    top_n: int,
    vec_repo: AbstractVectorRepo = Provide[DIContainer.vec_repo],
    text_model: AbstractEmbeddingModel = Provide[DIContainer.text_model],
    rerankers: Dict[Modal, AbstractReranker] = Provide[DIContainer.rerankers],
    storage: StorageClient = Provide[DIContainer.storage],
) -> KeysT:
    """
    1. For each element type, use the corresponding embedding model to
       embed text, use the embedding to query vec repo for the most
       relevant candidates (>= top_n candidates)
    2. Rerank candidates against query text using the reranker
       specific to the modal
    3. Return top_n elements
    """

    def _generate_objs(keys: Iterable[str]) -> Iterator[bytes]:
        for key in keys:
            yield storage[key]

    logger.info(f"Handling query text {user=} {text=}")
    res: KeysT = []
    query_vec = text_model.embed(text.encode("utf-8"))

    for modal in Modal:
        # query vector repo for candidates
        namespace = _get_vec_repo_namespace(user, modal)
        keys = vec_repo.query(namespace, query_vec, top_n * TOP_N_MULTIPLIER)
        logger.info(f"Found {len(keys)} candidates")

        if len(keys) < top_n:
            res.extend(keys)
            continue

        # rerank candidates
        reranker = rerankers[modal]
        ranks = reranker.rerank(text, _generate_objs(keys), top_n)
        res.extend([keys[i] for i in ranks])
    return res
