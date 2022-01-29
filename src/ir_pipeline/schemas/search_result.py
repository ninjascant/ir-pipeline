from typing import TypedDict, List


class SearchResultRow(TypedDict):
    qid: int
    query: str
    positive_doc_id: str
    hits: List[str]