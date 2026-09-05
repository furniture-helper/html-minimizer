import asyncio
import json

from kafka import KafkaProducer as KafkaProducerClient

from helpers.ecs_helpers import get_host, get_region


class KafkaService:

    def __init__(self, bootstrap_servers: str, topic: str):
        self.producer = KafkaProducerClient(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        )
        self.topic = topic
        self.metadata = {}

    def send_message(self, message: dict):
        if not self.metadata:
            self.metadata = {
                "host": asyncio.run(get_host()),
                "region": asyncio.run(get_region())
            }

        message_with_metadata = {**message, **self.metadata}
        self.producer.send(self.topic, value=message_with_metadata)

    def close(self):
        self.producer.flush()
        self.producer.close()