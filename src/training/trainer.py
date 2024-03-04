from typing import Callable
from src.settings import Settings

from datasets import Dataset
from transformers import Trainer, AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments

def _create_datacollator(tokenizer: AutoTokenizer) -> DataCollatorForTokenClassification:
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    return data_collator


def create_trainer(
        settings: Settings,
        model: AutoModelForTokenClassification,
        tokenizer: AutoTokenizer, 
        training_args: TrainingArguments, 
        compute_metrics: Callable, 
        dataset: Dataset
        ) -> Trainer:
        trainer = Trainer(
            model = model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            compute_metrics=compute_metrics,
            data_collator=_create_datacollator(tokenizer=tokenizer),
            tokenizer=tokenizer
        )
        return trainer