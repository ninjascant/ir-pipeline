from pathlib import Path
import click
from ir_pipeline.data_transformation.get_hybrid_results import HybridSearchRunner


@click.command()
@click.argument('versions', type=str)
@click.option('-d', '--dataset-name', type=str, default='msmarco-docs')
@click.option('-r', '--search-res-dir', type=str, default=None)
@click.option('-o', '--output-version', type=str, default=None)
def get_hybrid_res_command(versions, dataset_name, search_res_dir, output_version):
    versions = versions.split(',')
    if search_res_dir is None:
        search_res_dir = Path(Path.home(), '.ir-pipeline', dataset_name, 'search-results')
    transformer = HybridSearchRunner(search_res_dir)
    search_results = transformer.load_data(versions)
    hybrid_result = transformer.transform(search_results)
    if output_version is None:
        output_version = f'hybrid_res_{",".join(versions)}'
    transformer.save_data(Path(search_res_dir, output_version).with_suffix('.json'), hybrid_result)
