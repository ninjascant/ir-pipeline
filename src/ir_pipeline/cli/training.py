import click
from ..train.train_model import CustomModelTrainer


@click.command()
@click.argument('model_name_or_path', type=str)
@click.argument('train_set_dir', type=str)
@click.option('-c', '--checkpoint-dir', type=str, default='model_out/')
@click.option('-r', '--resume-training', type=bool, default=False)
@click.option('-f', '--fp16', type=bool, default=False)
def train_model_command(model_name_or_path, train_set_dir, checkpoint_dir, resume_training, fp16):
    trainer = CustomModelTrainer(
        model_name_or_path=model_name_or_path,
        checkpoint_dir=checkpoint_dir,
        resume_training=resume_training,
    )

    trainer.train_model(train_set_dir, fp16=fp16)
