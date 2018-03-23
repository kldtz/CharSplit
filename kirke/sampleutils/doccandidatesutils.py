from abc import ABC, abstractmethod
from typing import Dict, List

class DocCandidatesTransformer:

    def __init__(self) -> None:
        self.label = 'span_default_base'

    @abstractmethod        
    def enrich(self, candidate: Dict) -> None:
        pass
    
    # we structured postproc to take a list of candidates for a doc
    # because we want to handle the case when we only want the first
    # date in a document and reject all other dates.
    def doc_postproc(self, candidates: List[Dict]) -> None:
        for candidate in candidates:
            self.enrich(candidate)
            
