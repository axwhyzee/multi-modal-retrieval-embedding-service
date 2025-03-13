import logging

from event_core.adapters.pubsub import RedisConsumer
from event_core.domain.events import ElementStored

from bootstrap import bootstrap
from services.handlers import handle_element

logger = logging.getLogger(__name__)


def main():
    logger.info("Listening to event broker")
    with RedisConsumer() as consumer:
        consumer.subscribe(ElementStored)
        consumer.listen(handle_element)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bootstrap()
    main()
