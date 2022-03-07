from pathlib import Path
from typing import Optional, Union
from transformers import Trainer, TrainingArguments
from datasets import Dataset, load_from_disk
from ..models.triplet_bert import TripletBert


class TrainingArgs:
    def __init__(
            self,
            num_epochs: int = 1,
            batch_size: int = 8,
            accumulation_steps: int = 1,
            lr: float = 1e-5,
            logging_steps: int = 500,
            save_strategy: str = 'epoch',

    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.lr = lr

        self.logging_steps = logging_steps
        self.save_strategy = save_strategy


class CustomModelTrainer:
    def __init__(self,
                 model_name_or_path: str,
                 checkpoint_dir: Optional[str] = None,
                 resume_training: bool = False,
                 **kwargs):
        self.training_args = TrainingArgs(**kwargs)
        self.model_name_or_path = model_name_or_path
        self.checkpoint_dir = checkpoint_dir
        self.resume_training = resume_training

    @staticmethod
    def _load_data(dataset_dir: Union[Path, str]) -> Dataset:
        dataset = load_from_disk(dataset_dir)
        return dataset

    def train_model(self, dataset_dir: Union[Path, str]):
        dataset = self._load_data(dataset_dir)
        print(dataset)
        model = TripletBert(self.model_name_or_path)
        train_args = TrainingArguments(
            lr_scheduler_type='constant',
            evaluation_strategy='no',
            warmup_steps=500,
            weight_decay=0.01,
            num_train_epochs=self.training_args.num_epochs,
            per_device_train_batch_size=self.training_args.batch_size,
            gradient_accumulation_steps=self.training_args.accumulation_steps,
            learning_rate=self.training_args.lr,
            save_strategy=self.training_args.save_strategy,
            logging_steps=500,
            output_dir=self.checkpoint_dir
        )
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=dataset['train']
        )
        trainer.train(
            resume_from_checkpoint=self.resume_training
        )
