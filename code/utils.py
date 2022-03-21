import json
import os
import torch
import random
import numpy as np

from datasets import load_metric


def read_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def compute_metrics(pred):
    pearson = load_metric("pearsonr").compute
    references = pred.label_ids
    predictions = pred.predictions
    metric = pearson(predictions=predictions, references=references)
    return metric