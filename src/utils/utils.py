import os
import shutil
import zipfile
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def unzip_data(zip_file_path: str, path_to_extract: str) ->  str:
    """
    Unzips a zip file
    """
    
    print(f"Extracting {zip_file_path} to {path_to_extract}")
    
    if not os.path.exists(path_to_extract) and zipfile.is_zipfile(zip_file_path):
        
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(path=path_to_extract)
            
    return os.path.join(path_to_extract, "NER_Dataset.csv")
        
                
def unzip_data(zip_file_path: str, path_to_extract: str) ->  str:
    """
    Unzips a zip file

    Parameters
    ----------
    zip_file_path : str
        Path to the zip file
    path_to_extract : str
        Path to the directory where the zip file should be extracted

    Returns
    -------
    str
        Path to the unzipped file

    """
    print(f"Extracting {zip_file_path} to {path_to_extract}")

    if not os.path.exists(path_to_extract) and zipfile.is_zipfile(zip_file_path):

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(path=path_to_extract)

    return os.path.join(path_to_extract, "NER_Dataset.csv")


def extract_data_to_json(data_read_path: str, data_write_path: str) -> None:
    """
    Extracts data from a CSV file and saves it as JSON

    Parameters
    ----------
    data_read_path : str
        Path to the CSV file containing the data
    data_write_path : str
        Path to the directory where the JSON files should be saved

    """
    assert os.path.exists(data_read_path), "Data path does not exist!"

    df = pd.read_csv(data_read_path)[["Word", "Tag"]]

    if os.path.exists(data_write_path):
        os.remove(data_write_path)
    else:
        os.makedirs(data_write_path)

    with open(os.path.join(data_write_path, "data.json"), 'w') as outfile:
        for inputs, ner_tags in zip(df["Word"], df["Tag"]):
            outfile.write(json.dumps({"inputs": inputs, "ner_tags": ner_tags}) + "\n")
    
    
def unzip_data(zip_file_path: str, path_to_extract: str) ->  str:
    """
    Unzips a zip file

    Parameters
    ----------
    zip_file_path : str
        Path to the zip file
    path_to_extract : str
        Path to the directory where the zip file should be extracted

    Returns
    -------
    str
        Path to the unzipped file

    """
    print(f"Extracting {zip_file_path} to {path_to_extract}")

    if not os.path.exists(path_to_extract) and zipfile.is_zipfile(zip_file_path):

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(path=path_to_extract)

    return os.path.join(path_to_extract, "NER_Dataset.csv")


def extract_data_to_json(data_read_path: str, data_write_path: str) -> str:
    """
    Extracts data from a CSV file and saves it as JSON
    """
    assert os.path.exists(data_read_path), "Data path does not exist!"

    df = pd.read_csv(data_read_path)[["Word", "Tag"]]

    if os.path.exists(data_write_path):
        shutil.rmtree(data_write_path)
    else:
        os.makedirs(data_write_path)
        
    json_dataset_path = os.path.join(data_write_path, "data.json")

    with open(json_dataset_path, 'w') as outfile:
        for inputs, ner_tags in zip(df["Word"], df["Tag"]):
            outfile.write(json.dumps({"inputs": inputs, "ner_tags": ner_tags}) + "\n")
            
    return json_dataset_path


def load_dataset_from_json(data_read_path: str) -> Dataset:
    """
    Loads a Hugging Face Dataset from a JSON file
    """
    assert os.path.exists(data_read_path), "Data path does not exist!"
    
    return load_dataset("json", data_files=data_read_path)


def extract_unique_list_of_tags(dataset: Dataset) -> list:
    """
    Extracts a list of unique tags from a Hugging Face Dataset
    """
    
    all_tags = set()
    for sample in dataset["train"]:
        all_tags.update(set(sample["ner_tags"]))
        
    return sorted(list(all_tags), key= lambda x: (x[2:], x[0]))


def prepare_dataset_for_ner(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:  
    unique_list_of_tags = extract_unique_list_of_tags(dataset)
    ner_tag_to_numeric_label = {ner_tag: label for label, ner_tag in enumerate(unique_list_of_tags)}
    begin_to_inside = {k: k+1 for k in range(1, len(unique_list_of_tags), 2)}
    
    def align_targets(labels, word_ids):
        aligned_labels = []
        last_word = None
        for word in word_ids:
            if word is None:
                label = -100
            elif word == last_word:
                label = labels[word]
            else:
                label = labels[word]
                if label in begin_to_inside:
                    label = begin_to_inside[label]
            aligned_labels.append(label)
            last_word = word
        return aligned_labels
    
    def convert_tags_to_numeric_labels(tags: list, ner_tag_to_numeric_label: dict):
        return [ner_tag_to_numeric_label[tag] for tag in tags]
    
    def tokenize_fn(batch):
        tokenized_inputs = tokenizer(batch["inputs"], truncation=True, is_split_into_words=True)
        labels_batch = [convert_tags_to_numeric_labels(tags=tags, ner_tag_to_numeric_label=ner_tag_to_numeric_label) for tags in batch['ner_tags']]
        aligned_labels_batch = []
        for i, labels in enumerate(labels_batch):
            word_ids = tokenized_inputs.word_ids(i)
            aligned_labels_batch.append(align_targets(labels=labels, word_ids=word_ids))
        tokenized_inputs["labels"] = aligned_labels_batch
        return tokenized_inputs
    
    dataset = dataset["train"].train_test_split(seed=42)
    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)
    return tokenized_dataset
    
    
        
    
    
    
           

