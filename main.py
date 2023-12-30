# %%
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pandas as pd

from pathlib import Path
from src.make_dataset import encode, get_batches
from src.model import SimpleLlama
from src.train import Train
from src.infer import generate



# Define the path to the file
file_path = Path(__file__).parent / 'data/input/input.txt'
lines = open(file_path, 'r').read()
vocab = sorted(list(set(lines)))

MASTER_CONFIG = {
    "vocab_size": len(vocab),
    'batch_size': 4,
    'context_window': 16,
    'd_model': 128,
    'epochs': 1000,
    'log_interval': 10,
    'batch_size': 32,
    'n_heads': 8,
} 


dataset = torch.tensor(encode(lines, vocab), dtype=torch.int8)
model = SimpleLlama(config=MASTER_CONFIG)

xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])
logits, loss = model(xs, ys)
print(logits.shape, loss.shape)

optimizer = torch.optim.AdamW(
    model.parameters(), 
)

losses = Train(model, dataset, optimizer, config=MASTER_CONFIG).pre_train()

losses.plot()

# print(generate(model, vocab, config=MASTER_CONFIG))


# %%
