from abc import (
    ABC,
    abstractmethod,
)

from event_channel.exception import EndOfChannel


class BaseHandler(ABC):

    async def __aenter__(self):
        return self


class ReadHandler(BaseHandler):
    reader = None

    @abstractmethod
    async def read(self):
        ...

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await self.read()
        except EndOfChannel:
            raise StopAsyncIteration


class WriteHandler(BaseHandler):
    writer = None

    @abstractmethod
    async def write(self, value):
        ...

