from typing import List
from typing_extensions import TypedDict


class SearchResultRow(TypedDict):
    qid: int
    query: str
    positive_doc_id: str
    hits: List[str]