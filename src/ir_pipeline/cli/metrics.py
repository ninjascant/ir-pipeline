import click
from ir_pipeline.metrics.mrr import compute_mrr
from ir_pipeline.schemas.search_result import SearchResultRow
from ir_pipeline.utils.file_utils import read_jsonl_file


@click.command()
@click.argument('search_res_file', type=str)
def compute_mrr_command(search_res_file):
    search_results = read_jsonl_file(search_res_file)
    search_results = [SearchResultRow(**row) for row in search_results]
    compute_mrr(search_results)