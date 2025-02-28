import logging

from event_core.adapters.pubsub import RedisConsumer
from event_core.domain.events import ChunkStored

from bootstrap import bootstrap
from services.handlers import handle_chunk

logger = logging.getLogger(__name__)


def main():
    with RedisConsumer() as consumer:
        consumer.subscribe(ChunkStored)
        consumer.listen(handle_chunk)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bootstrap()
    main()
