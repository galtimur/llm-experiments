import os

import pandas as pd
from datasets import Dataset, load_dataset

datapath = "/mnt/data2/huggingface/datasets"
dataset = load_dataset("bookcorpus", cache_dir=datapath)

merged_dataset = []
buffer = ""
buffer_length = 0
for item in dataset["train"]:
    # Add the input example into the buffer.
    buffer += item["text"] + " "
    buffer_length += len(item["text"])

    if buffer_length >= 2100:
        merged_dataset.append(buffer)
        buffer = ""
        buffer_length = 0

    if len(merged_dataset) >= 540:
        break

df = pd.DataFrame({"text": merged_dataset})
val_dataset = Dataset.from_pandas(df)
items = [item["text"] for item in val_dataset]
lengths = [len(item["text"].split(" ")) for item in val_dataset]
print(sum(lengths) / len(lengths))

split_path = os.path.join(datapath, "bookcorpus", "splits", "val.json")
val_dataset.to_json(split_path)
