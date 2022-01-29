from typing import List
from loguru import logger
import numpy as np
from ..schemas.search_result import SearchResultRow


def compute_reciprocal_rank(row: SearchResultRow):
    if row['positive_doc_id'] in row['hits']:
        position = row['hits'].index(row['positive_doc_id']) + 1
        reciprocal_rank = 1 / position
    else:
        reciprocal_rank = 0
    return reciprocal_rank


def compute_mrr(search_results: List[SearchResultRow]):
    num_hits = len(search_results[0]['hits'])
    ranks = [compute_reciprocal_rank(row) for row in search_results]
    mrr = np.mean(ranks)
    logger.info(f'MRR@{num_hits}: {"%.5f" % mrr}')
    return mrr
