from transformers import TrainingArguments

from src.settings import Settings


def create_training_args(settings: Settings):
    training_args = TrainingArguments(
        output_dir=settings.model_name,
        evaluation_strategy=settings.evaluation_strategy,
        save_strategy=settings.save_strategy,
        learning_rate=settings.learning_rate,
        num_train_epochs=settings.num_train_epochs,
        weight_decay=settings.weight_decay,
        fp16=settings.fp16,
        fp16_backend=settings.fp16_backend,
        seed=settings.seed,
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=settings.batch_size,
    )
    return training_args
