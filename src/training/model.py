from transformers import  AutoModelForTokenClassification
from src.settings import Settings


def create_model(settings: Settings, id_to_label: dict, label_to_id: dict) -> AutoModelForTokenClassification:
    """
    Creates a Hugging Face model for token classification
    """
    model: AutoModelForTokenClassification = AutoModelForTokenClassification.from_pretrained(
        settings.checkpoint, 
        id2label=id_to_label, 
        label2id=label_to_id
        )
    return model
