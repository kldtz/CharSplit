from abc import abstractmethod
from typing import Dict


# pylint: disable=too-few-public-methods
class DocCandidatesTransformer:

    def __init__(self) -> None:
        self.label = 'span_default_base'

    @abstractmethod
    def enrich(self, candidate: Dict) -> None:
        pass
