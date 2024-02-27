from src.utils import unzip_data, extract_data_to_json, load_dataset_from_json, prepare_dataset_for_ner
from transformers import AutoTokenizer

CHECKPOINT = "distilbert-base-cased"

ZIP_RAW_DATA_FULL_PATH = "./data/archive.zip"
CSV_RAW_DATA_DIR_PATH = "./data/data_csv"
JSON_DATA_DIR_PATH = "./data/data_json"


if __name__ == "__main__":
    csv_raw_data_full_path: str = unzip_data(ZIP_RAW_DATA_FULL_PATH, CSV_RAW_DATA_DIR_PATH)
    json_datast_full_path = extract_data_to_json(data_read_path=csv_raw_data_full_path, data_write_path=JSON_DATA_DIR_PATH)
    dataset = load_dataset_from_json(json_datast_full_path)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    ner_dataset = prepare_dataset_for_ner(dataset=dataset, tokenizer=tokenizer)