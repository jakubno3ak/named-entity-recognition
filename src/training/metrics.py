from src.dataset import DatasetMetadata
from src.settings import Settings
from typing import Callable
from datasets import load_metric
from datasets import Metric

import numpy as np


def create_compute_metrics_callable(settings: Settings, metadata: DatasetMetadata) -> Callable:
    metric: Metric = load_metric(settings.metric)
    label_names: list = metadata.label_names

    def compute_metrics(logits_and_labels) -> dict:
        logits, labels = logits_and_labels
        preds = np.argmax(logits, axis=-1)
        str_labels = [[label_names[t] for t in label if t != -100] for label in labels]
        str_preds = [[label_names[p] for p, t  in zip(pred, targ) if t != -100] for pred, targ in zip(preds, labels)]
        the_metrics = metric.compute(predictions=str_preds, references=str_labels)
        return {
            "precision": the_metrics['overall_precision'],
            "recall": the_metrics['overall_recall'],
            "f1": the_metrics['overall_f1'],
            "accuracy": the_metrics['overall_accuracy']
        }
    
    return compute_metrics

