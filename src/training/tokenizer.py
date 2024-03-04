from transformers import AutoTokenizer
from src.settings import Settings

def create_tokenizer(settings: Settings) -> AutoTokenizer:
    """
    Creates a Hugging Face tokenizer
    """

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(settings.checkpoint)
    return tokenizer