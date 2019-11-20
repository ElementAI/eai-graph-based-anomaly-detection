from abc import ABC, abstractmethod
from torch import Tensor


class Aggregator(ABC):

    def __init__(self) -> None:
        # Produce a sample list of features and take its length
        self._entries = len(self(Tensor([0])))

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        pass

    def entries(self) -> int:
        return self._entries


class SimpleAggregator(Aggregator):

    def __call__(self, x: Tensor) -> Tensor:
        return [x.sum(), x.mean()]
