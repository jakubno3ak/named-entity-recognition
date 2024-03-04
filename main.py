from src.dataset import create_dataset
from src.settings import Settings
from src.training import (
    create_compute_metrics_callable,
    create_model,
    create_tokenizer,
    create_trainer,
    create_training_args,
)


def main():
    settings = Settings()
    tokenizer = create_tokenizer(settings=settings)
    dataset, metadata = create_dataset(settings=settings, tokenizer=tokenizer)
    model = create_model(
        settings=settings,
        id_to_label=metadata.id_to_label,
        label_to_id=metadata.label_to_id,
    )
    compute_metrics = create_compute_metrics_callable(
        settings=settings, metadata=metadata
    )
    training_args = create_training_args(settings=settings)
    trainer = create_trainer(
        settings=settings,
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        compute_metrics=compute_metrics,
        dataset=dataset,
    )

    trainer.train()


if __name__ == "__main__":
    main()
