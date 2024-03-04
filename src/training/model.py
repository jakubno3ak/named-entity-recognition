import torch
from transformers import AutoModelForTokenClassification

from src.settings import Device, Settings


def create_model(
    settings: Settings, id_to_label: dict, label_to_id: dict
) -> AutoModelForTokenClassification:
    """
    Creates a Hugging Face model for token classification
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() and settings.device == Device.GPU else "cpu"
    )
    model: AutoModelForTokenClassification = (
        AutoModelForTokenClassification.from_pretrained(
            settings.checkpoint, id2label=id_to_label, label2id=label_to_id
        ).to(device)
    )

    return model
