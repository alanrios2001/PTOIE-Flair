# Portuguese dataset
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from OIE.datasets.validated_splits.contractions import clean_extraction, transform_portuguese_contractions


@dataclass
class TripleExtraction:
    arg1: str
    rel: str
    arg2: str
    valid: bool = True

    def to_dict(self):
        return {
            "arg1": self.arg1,
            "rel": self.rel,
            "arg2": self.arg2,
            "valid": self.valid,
            "confidence": 1.0,
        }

    def __str__(self):
        return f'arg1:{self.arg1} - rel:{self.rel} - arg2:{self.arg2} - valid:{self.valid}'


@dataclass
class GenerativeSentence:
    phrase: str
    gold_extractions: List[TripleExtraction]
    predicted_extractions: List[TripleExtraction] = field(default_factory=lambda: [])


def load_pragmatic_wiki_dataset():
    dataset = dict()

    pt = Path(__file__).parent / "pragmatic_dataset" / "wiki200.txt"
    with open(pt, "r", encoding="utf-8") as f_pt:
        for line in f_pt:
            line = line.strip()
            pos, phrase = line.split("\t", 1)
            phrase = clean_extraction(phrase)
            dataset[int(pos)] = {
                "phrase": transform_portuguese_contractions(phrase.strip()),
                "extractions": [],
            }

    pt = Path(__file__).parent / "pragmatic_dataset" / "wiki200-labeled.csv"
    with open(pt, "r", encoding="utf-8") as f_pt:
        for line in f_pt:
            if "\t" in line:
                partes = line.split("\t")
                pos = int(partes[0])
                arg1 = clean_extraction(partes[1])
                rel = clean_extraction(partes[2])
                arg2 = clean_extraction(partes[3])
                valid = partes[-1].strip()

                if valid != "1":
                    continue  # so queremos pegar as positivas

                dataset[pos]["extractions"].append(
                    {
                        "arg1": transform_portuguese_contractions(arg1),
                        "rel": transform_portuguese_contractions(rel),
                        "arg2": transform_portuguese_contractions(arg2),
                        "valid": int(valid),
                    }
                )
    return dataset


def load_pragmatic_ceten_dataset():
    dataset_pt = dict()

    pt = Path(__file__).parent / "pragmatic_dataset" / "ceten200.txt"
    with open(pt, "r", encoding="utf-8") as f_pt:
        for line in f_pt:
            line = line.strip()
            pos, phrase = line.split("\t", 1)
            phrase = clean_extraction(phrase)
            dataset_pt[int(pos)] = {
                "phrase": transform_portuguese_contractions(phrase.strip()),
                "extractions": [],
            }

    pt = Path(__file__).parent / "pragmatic_dataset" / "ceten200-labeled.csv"
    with open(pt, "r", encoding="utf-8") as f_pt:
        for line in f_pt:
            if "\t" in line:
                partes = line.split("\t")
                pos = int(partes[0])
                arg1 = clean_extraction(partes[1])
                rel = clean_extraction(partes[2])
                arg2 = clean_extraction(partes[3])
                valid = partes[-1].strip()

                if valid != "1":
                    continue  # so queremos pegar as positivas

                dataset_pt[pos]["extractions"].append(
                    {
                        "arg1": transform_portuguese_contractions(arg1),
                        "rel": transform_portuguese_contractions(rel),
                        "arg2": transform_portuguese_contractions(arg2),
                        "valid": int(valid),
                    }
                )
    return dataset_pt


def load_pud200():
    # Portuguese dataset
    dataset_pt = dict()
    pt = Path(__file__).parent / "normal/eval" / "200-sentences-pt-PUD.txt"
    with open(pt, "r", encoding="utf-8") as f_pt:
        actual_pos = None
        for line in f_pt:
            line = line.strip()
            pos, phrase = line.split("\t", 1)
            phrase = clean_extraction(phrase)

            if pos.isnumeric() and phrase.count("\t") < 1:
                actual_pos = int(pos)
                phrase = transform_portuguese_contractions(phrase)
                # phrase = re.sub(r',|\.|"', "", phrase)
                dataset_pt[actual_pos] = {"phrase": phrase, "extractions": []}
            else:
                partes = line.split("\t")
                arg1 = clean_extraction(partes[0])
                rel = clean_extraction(partes[1])
                arg2 = clean_extraction(partes[2])
                valid = int(partes[-2].strip())

                if valid != 1:
                    continue  # so queremos pegar as positivas

                dataset_pt[actual_pos]["extractions"].append(
                    {
                        "arg1": transform_portuguese_contractions(arg1),
                        "rel": transform_portuguese_contractions(rel),
                        "arg2": transform_portuguese_contractions(arg2),
                        "valid": valid,
                    }
                )

    return dataset_pt


def load_pud100():
    # Portuguese dataset
    dataset_pt = dict()
    pt = Path(__file__).parent / "normal/eval" / "coling2020.txt"
    with open(pt, "r", encoding="utf-8") as f_pt:
        actual_pos = None
        for line in f_pt:
            line = line.strip()
            pos, phrase = line.split("\t", 1)
            # phrase = clean_extraction(phrase)

            if pos.isnumeric() and phrase.count("\t") < 1:
                actual_pos = int(pos)
                # phrase = transform_portuguese_contractions(phrase)
                # phrase = re.sub(r',|\.|"', "", phrase)
                dataset_pt[actual_pos] = {"phrase": phrase, "extractions": []}
            else:
                partes = line.split("\t")
                arg1 = clean_extraction(partes[0])
                rel = clean_extraction(partes[1])
                arg2 = clean_extraction(partes[2])
                valid = int(partes[-2].strip())

                if valid != 1:
                    continue  # so queremos pegar as positivas

                dataset_pt[actual_pos]["extractions"].append(
                    {
                        "arg1": transform_portuguese_contractions(arg1),
                        "rel": transform_portuguese_contractions(rel),
                        "arg2": transform_portuguese_contractions(arg2),
                        "valid": valid,
                    }
                )

    return dataset_pt


def load_anderson():
    files_path = Path(f"../../datasets/anderson").resolve()
    files = files_path.glob("*")

    dataset_pt = dict()
    actual_pos = 0

    for pt in files:
        with open(pt, "r", encoding="utf-8") as f_pt:
            for line in f_pt:
                result = json.loads(line)

                pos, phrase = line.split("\t", 1)
                # phrase = clean_extraction(phrase)

                if pos.isnumeric() and phrase.count("\t") < 1:
                    actual_pos = int(pos)
                    # phrase = transform_portuguese_contractions(phrase)
                    # phrase = re.sub(r',|\.|"', "", phrase)
                    dataset_pt[actual_pos] = {"phrase": phrase, "extractions": []}
                else:
                    partes = line.split("\t")
                    arg1 = clean_extraction(partes[0])
                    rel = clean_extraction(partes[1])
                    arg2 = clean_extraction(partes[2])
                    valid = int(partes[-2].strip())

                    if valid != 1:
                        continue  # so queremos pegar as positivas

                    dataset_pt[actual_pos]["extractions"].append(
                        {
                            "arg1": transform_portuguese_contractions(arg1),
                            "rel": transform_portuguese_contractions(rel),
                            "arg2": transform_portuguese_contractions(arg2),
                            "valid": valid,
                        }
                    )

    return dataset_pt


def load_gamalho():
    # Portuguese dataset
    dataset_pt = dict()

    pt = Path(__file__).parent / "gamalho" / "sentences.txt"
    with open(pt, "r", encoding="utf-8") as f_pt:
        for line in f_pt:
            line = line.strip()
            pos, phrase = line.split("\t", 1)
            phrase = clean_extraction(phrase)
            dataset_pt[int(pos)] = {
                "phrase": transform_portuguese_contractions(phrase),
                "extractions": [],
            }

    pt = Path(__file__).parent / "gamalho" / "gold.csv"

    def clean_at_symbol(text):
        text = text.replace("@", " ")
        return text

    with open(pt, "r", encoding="utf-8") as f_pt:
        for line in f_pt:
            if "\t" in line:
                partes = line.split("\t")
                pos = int(partes[0])
                arg1 = clean_at_symbol(clean_extraction(partes[1]))
                rel = clean_at_symbol(clean_extraction(partes[2]))
                arg2 = clean_at_symbol(clean_extraction(partes[3]))
                if len(partes[-1].strip()) < 1:
                    continue
                valid = int(partes[-1].strip())

                if valid != 1:
                    continue  # so queremos pegar as positivas

                dataset_pt[pos]["extractions"].append(
                    {
                        "arg1": transform_portuguese_contractions(arg1),
                        "rel": transform_portuguese_contractions(rel),
                        "arg2": transform_portuguese_contractions(arg2),
                        "valid": valid,
                    }
                )

    return dataset_pt


def load_gen2oie():
    # Portuguese dataset
    dataset_pt = dict()
    sentence_to_pos = dict()

    pt = Path(__file__).parent / "gen2oie" / "PT" / "s2_train.tsv"

    with open(pt, "r", encoding="utf-8") as f_pt:
        actual_pos = 0
        for line in f_pt:
            partes = line.split("\t")

            sentence = partes[0].split("<r>")[1].strip()
            extraction = partes[1]

            arg1_regex = "<a1>([^<]*)<"
            arg2_regex = "<a2>([^<]*)<"
            rel_regex = "<r>([^<]*)<"

            arg1 = re.search(arg1_regex, extraction)
            arg2 = re.search(arg2_regex, extraction)
            rel = re.search(rel_regex, extraction)

            if not arg1 or not arg2 or not rel:
                continue

            arg1 = arg1.group(1).strip()
            arg2 = arg2.group(1).strip()
            rel = rel.group(1).strip()

            if sentence not in sentence_to_pos:
                sentence_to_pos[sentence] = actual_pos
                dataset_pt[actual_pos] = {"phrase": sentence, "extractions": []}
                actual_pos += 1

            pos = sentence_to_pos[sentence]

            dataset_pt[pos]["extractions"].append(
                {
                    "arg1": transform_portuguese_contractions(arg1),
                    "rel": transform_portuguese_contractions(rel),
                    "arg2": transform_portuguese_contractions(arg2),
                    "valid": 1,
                }
            )

    return dataset_pt


def load_alan_train():
    pt = "validated_splits" / "lsoie" / "train_valid.json"
    dataset_pt = dict()
    with open(pt, "r", encoding="utf-8") as f_pt:
        actual_pos = 0
        dataset = json.load(f_pt)

        for sentence in dataset:
            dataset_pt[actual_pos] = {"phrase": sentence, "extractions": []}
            arg1 = " ".join(dataset[sentence]["arg1"])
            arg2 = " ".join(dataset[sentence]["arg2"])
            rel = " ".join(dataset[sentence]["rel"])

            dataset_pt[actual_pos]["extractions"].append(
                {
                    "arg1": transform_portuguese_contractions(arg1),
                    "rel": transform_portuguese_contractions(rel),
                    "arg2": transform_portuguese_contractions(arg2),
                    "valid": 1,
                }
            )

            actual_pos += 1
        return dataset_pt

def load_alan_test():
    pt = Path(__file__).parent / "alan" / "test_valid.json"
    dataset_pt = dict()
    with open(pt, "r", encoding="utf-8") as f_pt:
        actual_pos = 0
        dataset = json.load(f_pt)

        for sentence in dataset:
            dataset_pt[actual_pos] = {"phrase": sentence, "extractions": []}
            arg1 = " ".join(dataset[sentence]["arg1"])
            arg2 = " ".join(dataset[sentence]["arg2"])
            rel = " ".join(dataset[sentence]["rel"])

            dataset_pt[actual_pos]["extractions"].append(
                {
                    "arg1": transform_portuguese_contractions(arg1),
                    "rel": transform_portuguese_contractions(rel),
                    "arg2": transform_portuguese_contractions(arg2),
                    "valid": 1,
                }
            )

            actual_pos += 1
        return dataset_pt

def load_alan_dev():
    pt = Path(__file__).parent / "alan" / "dev_valid.json"
    dataset_pt = dict()
    with open(pt, "r", encoding="utf-8") as f_pt:
        actual_pos = 0
        dataset = json.load(f_pt)

        for sentence in dataset:
            dataset_pt[actual_pos] = {"phrase": sentence, "extractions": []}
            arg1 = " ".join(dataset[sentence]["arg1"])
            arg2 = " ".join(dataset[sentence]["arg2"])
            rel = " ".join(dataset[sentence]["rel"])

            dataset_pt[actual_pos]["extractions"].append(
                {
                    "arg1": transform_portuguese_contractions(arg1),
                    "rel": transform_portuguese_contractions(rel),
                    "arg2": transform_portuguese_contractions(arg2),
                    "valid": 1,
                }
            )

            actual_pos += 1
        return dataset_pt

def load_alan_gold():
    pt = Path(__file__).parent / "alan" / "gold_valid.json"
    dataset_pt = dict()
    with open(pt, "r", encoding="utf-8") as f_pt:
        actual_pos = 0
        dataset = json.load(f_pt)

        for sentence in dataset:
            dataset_pt[actual_pos] = {"phrase": sentence, "extractions": []}
            arg1 = " ".join(dataset[sentence]["arg1"])
            arg2 = " ".join(dataset[sentence]["arg2"])
            rel = " ".join(dataset[sentence]["rel"])

            dataset_pt[actual_pos]["extractions"].append(
                {
                    "arg1": transform_portuguese_contractions(arg1),
                    "rel": transform_portuguese_contractions(rel),
                    "arg2": transform_portuguese_contractions(arg2),
                    "valid": 1,
                }
            )

            actual_pos += 1
        return dataset_pt


def get_dataset():
    datasets = [
        #{"name": "pragmatic_wiki", "dataset": load_pragmatic_wiki_dataset(), "type": "train"},
        #{"name": "pragmatic_ceten", "dataset": load_pragmatic_ceten_dataset(), "type": "train"},
        #{"name": "gamalho", "dataset": load_gamalho(), "type": "train"},
        {"name": "pud_200", "dataset": load_pud200(), "type": "train"},
        {"name": "pud_100", "dataset": load_pud100(), "type": "test"},
        #{"name": "alan_gold", "dataset": load_alan_gold(), "type": "train"},
        #{"name": "alan_train", "dataset": load_alan_train(), "type": "train"},
        #{"name": "alan_test", "dataset": load_alan_test(), "type": "train"},
        #{"name": "alan_dev", "dataset": load_alan_dev(), "type": "train"},
        #{"name": "gen2oie", "dataset": load_gen2oie(), "type": "train"},
        # {"name": "anderson", "dataset": load_anderson(), "type": "train"},
    ]

    actual_pos = 0

    train_dataset = []
    test_dataset = []

    for dataset_to_process in datasets:
        extractions = dataset_to_process["dataset"]

        list_extractions = []
        for pos_sentence, value in enumerate(extractions.values()):
            sentence_extractions = []
            for triple in value["extractions"]:
                sentence_extractions.append(
                    TripleExtraction(
                        arg1=triple["arg1"],
                        rel=triple["rel"],
                        arg2=triple["arg2"],
                        valid=triple["valid"],
                    )
                )

            list_extractions.append(
                GenerativeSentence(phrase=value["phrase"], gold_extractions=sentence_extractions)
            )

        if dataset_to_process["type"] == "train":
            train_dataset.extend(list_extractions)
        else:
            test_dataset.extend(list_extractions)

    return train_dataset, test_dataset


if __name__ == "__main__":
    dataset = get_dataset()
    #print(dataset)
