import logging
from typing import Dict, Iterable, Iterator, List, Optional, Type, TypeAlias

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


def _get_vec_repo_idx_name(elem: Element) -> str:
    return elem.value


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
    event_cls = event.__class__
    event_elem = ELEM_TYPES[event_cls]
    model = model_factory[event_cls]

    try:
        vec = model.embed(storage[key])
    except:
        return

    vec_repo.insert(
        index_name=_get_vec_repo_idx_name(event_elem),
        namespace=_user_from_key(key),
        key=key,
        vec=vec,
    )


@inject
def handle_query_text(
    user: str,
    text: str,
    top_n: int,
    exclude_elems: Optional[List[str]] = None,
    storage: StorageClient = Provide[DIContainer.storage],
    vec_repo: AbstractVectorRepo = Provide[DIContainer.vec_repo],
    query_model_factory: Dict[Element, AbstractEmbeddingModel] = Provide[
        DIContainer.query_model_factory
    ],
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
    enc_text = text.encode("utf-8")

    for elem, text_model in query_model_factory.items():
        if exclude_elems and elem.value in exclude_elems:
            continue

        # query vector repo for candidates
        query_vec = text_model.embed(enc_text)
        keys = vec_repo.query(
            index_name=_get_vec_repo_idx_name(elem),
            namespace=user,
            vec=query_vec,
            top_k=top_n * TOP_N_MULTIPLIER,
        )
        logger.info(f"Found {len(keys)} candidates")

        if len(keys) < top_n:
            res.extend(keys)
            continue

        # rerank candidates
        if reranker := reranker_factory.get(elem):
            ranks = reranker.rerank(text, _generate_objs(keys), top_n)
            res.extend([keys[i] for i in ranks])
        else:
            res.extend(keys[:top_n])
    return res
