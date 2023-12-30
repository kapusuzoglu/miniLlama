import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pandas as pd

# simple tokenization by characters
def encode(s, vocab):
    stoi = {ch:i for i, ch in enumerate(vocab)}
    return [stoi[ch] for ch in s]

def decode(l, vocab):
    itos = {i:ch for i, ch in enumerate(vocab)}
    return ''.join([itos[i] for i in l])

def get_batches(data, split, batch_size, context_window, config=None):
    threshold_train = .8
    threshold_inc_val = .1
    data_train_lim = int(threshold_train * len(data))
    threshold_val = threshold_train + threshold_inc_val
    data__val_lim = int(threshold_val * len(data))
    
    train = data[:data_train_lim]
    val = data[data_train_lim: data__val_lim]
    test = data[data__val_lim:]
    
    batch_data = train
    if split == 'val':
        batch_data = val
    if split == 'test':
        batch_data = test
    
    # pick random starting points
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()
    return x, y
