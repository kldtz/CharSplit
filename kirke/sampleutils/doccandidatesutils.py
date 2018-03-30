from abc import ABC, abstractmethod
from typing import Dict, List

class DocCandidatesTransformer:

    def __init__(self) -> None:
        self.label = 'span_default_base'

    @abstractmethod        
    def enrich(self, candidate: Dict) -> None:
        pass
    
