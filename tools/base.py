from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    name: str
    description: str
    direct_response: bool = False

    @abstractmethod
    def run(self, query: str, app, **kwargs) -> tuple[str, list[Any]]:
        pass
