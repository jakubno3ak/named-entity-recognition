from enum import Enum

from pydantic_settings import BaseSettings


class Device(str, Enum):
    CPU = "cpu"
    GPU = "gpu"


class Settings(BaseSettings):
    model_name: str = "distilbert-finetuned-ner"
    checkpoint: str = "distilbert-base-uncased"
    zip_data_full_path: str = "./data/archive.zip"
    csv_data_dir_path: str = "./data/data_csv"
    json_data_dir_path: str = "./data/data_json"
    metric: str = "seqeval"

    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    learning_rate: float = 3e-5
    num_train_epochs: int = 10
    weight_decay: float = 0.01
    seed: int = 42
    batch_size: int = 16
    device: Device = Device.GPU
    fp16: bool = True
    fp16_backend: str = "amp"
