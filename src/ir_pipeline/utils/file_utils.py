import json
from typing import Union, List
from pathlib import Path


def read_jsonl_file(file_path: Union[str, Path]):
    with open(file_path) as file:
        data = [json.loads(line) for line in file]
    return data


def write_jsonl_file(file_path: Union[str, Path], data: List):
    with open(file_path, 'w') as outfile:
        outfile.write('\n'.join(json.dumps(row) for row in data) + '\n')
