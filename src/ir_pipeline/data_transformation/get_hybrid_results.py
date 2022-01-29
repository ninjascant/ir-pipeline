from typing import Union, List, Dict
from pathlib import Path
import json
from collections import defaultdict
from tqdm.auto import tqdm
from ..utils.file_utils import write_jsonl_file
from ..schemas.search_result import SearchResultRow


class HybridSearchRunner:
    """
    Loads search results from different models and constructs a hybrid result
    """
    def __init__(self, search_res_dir: Union[str, Path]):
        self.search_res_dir = search_res_dir
        self._ranks = None

    @staticmethod
    def _load_search_result_to_mapping(file_path: Union[str, Path]) -> Dict[int, SearchResultRow]:
        qid_to_hits = {}
        with open(file_path) as file:
            for line in file:
                row_dict = json.loads(line)
                qid_to_hits[row_dict['qid']] = SearchResultRow(**row_dict)
        return qid_to_hits

    def _get_hybrid_res(self, joined_hits: List[List[str]]) -> List[str]:
        hybrid_hits = defaultdict(int)
        for hits in joined_hits:
            hits = zip(hits, self._ranks)
            for item in hits:
                hybrid_hits[item[0]] += item[1]
        hybrid_hits = [(k, v) for k, v in hybrid_hits.items()]
        hybrid_hits = sorted(hybrid_hits, key=lambda x: x[1], reverse=True)
        hybrid_hits = hybrid_hits[:len(self._ranks)]
        hybrid_hits = [item[0] for item in hybrid_hits]
        return hybrid_hits

    def load_data(self, search_res_versions: List[str]) -> List[Dict[int, SearchResultRow]]:
        search_res_files = [Path(self.search_res_dir, version).with_suffix('.json')
                            for version in search_res_versions]
        search_results = [self._load_search_result_to_mapping(res_file)
                          for res_file in tqdm(search_res_files)]
        return search_results

    @staticmethod
    def save_data(out_path: Union[str, Path], search_result: List[SearchResultRow]):
        write_jsonl_file(out_path, search_result)

    def transform(self, search_results: List[Dict[int, SearchResultRow]]) -> List[SearchResultRow]:
        total_qids = [set(res.keys()) for res in search_results]
        total_qids = set.intersection(*total_qids)
        hit_num = len(list(search_results[0].values())[0]['hits'])
        joined_results = []
        self._ranks = [1 / (i+1) for i in range(hit_num)]
        print(len(self._ranks))
        for qid in total_qids:
            row = search_results[0][qid]
            total_hits = [item[qid]['hits'] for item in search_results]
            hybrid_hits = self._get_hybrid_res(total_hits)
            row['hits'] = hybrid_hits
            joined_results.append(row)
        return joined_results
