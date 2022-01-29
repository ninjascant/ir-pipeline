import click
from .data_transformation import get_hybrid_res_command
from .search import run_ann_search_command
from .metrics import compute_mrr_command


@click.group()
def run():
    pass


run.add_command(get_hybrid_res_command, 'get_hybrid_res')
run.add_command(run_ann_search_command, 'run_ann_search')
run.add_command(compute_mrr_command, 'compute_mrr')
