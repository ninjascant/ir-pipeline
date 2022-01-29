from typing import Union, Optional
from pathlib import Path
import json
from loguru import logger
import numpy as np
import pandas as pd
import faiss
from ..utils.file_utils import read_jsonl_file, write_jsonl_file


NANE_TO_METRIC = {
    'l2': faiss.METRIC_L2,
    'dot-product': faiss.METRIC_INNER_PRODUCT
}


def convert_ids_to_int(ids):
    is_str = type(ids[0]) is str
    if is_str:
        ids = [int(doc_id[1:]) for doc_id in ids]
    return ids


def convert_idx_to_id(indices):
    return [f'D{idx}' for idx in indices]


class AnnSearcher:
    def __init__(
            self,
            doc_embedding_dir: Union[str, Path]
    ):
        if Path(doc_embedding_dir, 'index_l2.ann').exists():
            self.index = self._load_index(doc_embedding_dir)
        else:
            self.index = self._build_index(doc_embedding_dir)
        self.embedding_size = None

    def _load_queries(self, query_file):
        queries = read_jsonl_file(query_file)
        queries = pd.DataFrame(queries)
        return queries

    @staticmethod
    def _load_embeddings(embedding_dir_path: Union[str, Path]):
        with open(Path(embedding_dir_path, 'ids.json')) as json_file:
            ids = json.load(json_file)

        ids = convert_ids_to_int(ids)
        ids = np.array(ids).astype(np.int64)

        embeddings = np.load(Path(embedding_dir_path, 'embeddings.npy')).astype(np.float32)
        return embeddings, ids

    def _build_index(
            self,
            embedding_dir_path: Union[str, Path],
            metric: str = 'l2',
            factory_string: Optional[str] = 'IDMap,Flat'
    ):
        logger.info('Loading embeddings')
        embeddings, ids = self._load_embeddings(embedding_dir_path)
        embedding_size = embeddings.shape[1]
        index = faiss.index_factory(embedding_size, factory_string, NANE_TO_METRIC[metric])

        index.add_with_ids(embeddings, ids)
        out_path = str(Path(embedding_dir_path, f'index_{metric}.ann'))
        logger.info(f'Writing index to {out_path}')
        faiss.write_index(index, out_path)
        return index

    @staticmethod
    def _load_index(
            embedding_dir_path: Union[str, Path],
    ):
        index_path = Path(embedding_dir_path, 'index_l2.ann')
        index_path = str(index_path)
        index = faiss.read_index(index_path)
        return index

    def search(
            self,
            query_embedding_dir: Union[str, Path],
            query_file: Union[str, Path],
            out_file: str,  top_n: int):
        query_embeddings, query_ids = self._load_embeddings(query_embedding_dir)
        queries = self._load_queries(query_file)

        logger.info('Starting search')
        distances, search_res = self.index.search(query_embeddings, top_n)
        logger.info('Finished search')

        search_res = [convert_idx_to_id(item) for item in search_res]

        search_res_df = pd.DataFrame(zip(query_ids, search_res))
        search_res_df.columns = ['qid', 'hits']

        search_res_df = search_res_df.merge(queries, on='qid', how='inner')
        search_res_df = search_res_df.rename(columns={'text': 'query'})
        search_results = search_res_df.to_dict(orient='records')

        write_jsonl_file(out_file, search_results)
