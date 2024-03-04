from dataclasses import dataclass

from datasets import Dataset
from transformers import AutoTokenizer

from src.settings import Settings
from src.utils import unzip_data, extract_data_to_json, load_dataset_from_json, extract_unique_list_of_tags, prepare_dataset_for_ner

@dataclass
class DatasetMetadata:
    label_names: list
    id_to_label: dict
    label_to_id: dict


def create_dataset(settings: Settings, tokenizer: AutoTokenizer) -> tuple[Dataset, DatasetMetadata]:
    csv_raw_data_full_path: str = unzip_data(settings.zip_data_full_path, settings.csv_data_dir_path)
    json_datast_full_path: str = extract_data_to_json(data_read_path=csv_raw_data_full_path, data_write_path=settings.json_data_dir_path)
    dataset: Dataset = load_dataset_from_json(json_datast_full_path)
    label_names: dict = extract_unique_list_of_tags(dataset)
    dataset = prepare_dataset_for_ner(dataset, tokenizer=tokenizer)

    id_to_label: dict = {id: label for id, label in enumerate(label_names)}
    label_to_id: dict = {label: id for id, label in  id_to_label.items()}

    metadata: DatasetMetadata = DatasetMetadata(
        label_names=label_names,
        id_to_label=id_to_label, 
        label_to_id=label_to_id
        )
    
    return dataset, metadata

