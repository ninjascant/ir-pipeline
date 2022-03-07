import click
from ..train.train_model import CustomModelTrainer


@click.command()
@click.argument('model_name_or_path', type=str)
@click.argument('train_set_dir', type=str)
@click.option('checkpoint_dir', type=str, default=None)
def train_model_command(model_name_or_path, train_set_dir, checkpoint_dir):
    trainer = CustomModelTrainer(
        model_name_or_path=model_name_or_path,
        checkpoint_dir=checkpoint_dir
    )

    trainer.train_model(train_set_dir)
