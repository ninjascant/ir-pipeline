import click
from .data_transformation import get_hybrid_res_command, create_train_set_command
from .search import run_ann_search_command
from .metrics import compute_mrr_command
from .tokenize_data import tokenize_train_set_command
from .training import train_model_command


@click.group()
def run():
    pass


run.add_command(get_hybrid_res_command, 'get_hybrid_res')
run.add_command(run_ann_search_command, 'run_ann_search')
run.add_command(compute_mrr_command, 'compute_mrr')
run.add_command(create_train_set_command, 'create_train_set')
run.add_command(tokenize_train_set_command, 'tokenize_train_set')
run.add_command(train_model_command, 'train_model')
