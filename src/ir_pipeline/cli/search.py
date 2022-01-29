import click
from ir_pipeline.search.ann_search import AnnSearcher


@click.command()
@click.argument('doc_embedding_dir', type=str)
@click.argument('query_embedding_dir', type=str)
@click.argument('query_file', type=str)
@click.argument('outfile', type=str)
@click.option('-t', '--top-n', type=int, default=200)
def run_ann_search_command(doc_embedding_dir, query_embedding_dir, query_file, outfile, top_n):
    searcher = AnnSearcher(doc_embedding_dir)
    searcher.search(
        query_embedding_dir=query_embedding_dir,
        query_file=query_file,
        out_file=outfile,
        top_n=top_n)
