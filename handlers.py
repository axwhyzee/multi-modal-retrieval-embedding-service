import logging
from typing import Dict, Iterable, Iterator, List, Type, TypeAlias

from dependency_injector.wiring import Provide, inject
from event_core.adapters.services.storage import StorageClient
from event_core.domain.events.elements import ELEM_TYPES, ElementStored
from event_core.domain.types import Element

from adapters.embedders.base import AbstractEmbeddingModel
from adapters.repository import AbstractVectorRepo
from adapters.rerankers.base import AbstractReranker
from bootstrap import DIContainer
from config import TOP_N_MULTIPLIER

logger = logging.getLogger(__name__)


KeysT: TypeAlias = List[str]


def _user_from_key(key: str) -> str:
    return key.split("/")[0]


def _get_vec_repo_namespace(user: str, elem: Element) -> str:
    return f"{user}__{elem}"


@inject
def handle_element(
    event: ElementStored,
    storage: StorageClient = Provide[DIContainer.storage],
    vec_repo: AbstractVectorRepo = Provide[DIContainer.vec_repo],
    model_factory: Dict[Type[ElementStored], AbstractEmbeddingModel] = Provide[
        DIContainer.model_factory
    ],
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
    model = model_factory[event.__class__]
    elem = ELEM_TYPES[event.__class__]
    vec = model.embed(storage[key])    
    namespace = _get_vec_repo_namespace(user, elem)
    vec_repo.insert(namespace, key, vec)


@inject
def handle_query_text(
    user: str,
    text: str,
    top_n: int,
    storage: StorageClient = Provide[DIContainer.storage],
    vec_repo: AbstractVectorRepo = Provide[DIContainer.vec_repo],
    text_model: AbstractEmbeddingModel = Provide[DIContainer.text_model],
    reranker_factory: Dict[Element, AbstractReranker] = Provide[
        DIContainer.reranker_factory
    ],
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

    for elem in Element:
        # query vector repo for candidates
        namespace = _get_vec_repo_namespace(user, elem)
        keys = vec_repo.query(namespace, query_vec, top_n * TOP_N_MULTIPLIER)
        logger.info(f"Found {len(keys)} candidates")

        if len(keys) < top_n:
            res.extend(keys)
            continue

        # rerank candidates
        if reranker := reranker_factory[elem]:
            ranks = reranker.rerank(text, _generate_objs(keys), top_n)
            res.extend([keys[i] for i in ranks])
        else:
            res.extend(keys[:top_n])
    return res
