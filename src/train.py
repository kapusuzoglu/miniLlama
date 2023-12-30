import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd

from .make_dataset import get_batches

class Train():
    def __init__(self, model, dataset, optimizer, config=None):
        self.model = model
        self.config = config
        self.dataset = dataset
        self.optimizer = optimizer

    def pre_train(self, scheduler=None, print_logs=False):
        losses = []
        start_time = time.time()
        for epoch in range(self.config['epochs']):
            self.optimizer.zero_grad()
            
            xs, ys = get_batches(self.dataset, 'train', self.config['batch_size'], self.config['context_window'])
            logits, loss = self.model(xs, targets=ys)
            loss.backward()
            self.optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            if epoch % self.config['log_interval'] == 0:
                batch_time = time.time() - start_time
                x = self.evaluate_loss()
                losses += [x]
                if print_logs:
                    print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (self.config['epochs'] - epoch)/self.config['log_interval'] :.3f}")
                start_time = time.time()

                if scheduler:
                    print("lr: ", scheduler.get_lr())

        print("validation loss: ", losses[-1]['val'])
        return pd.DataFrame(losses)

    @torch.no_grad()  # don't compute gradients for this function
    def evaluate_loss(self):
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = []
            for _ in range(10):
                xb, yb = get_batches(self.dataset, split, self.config['batch_size'], self.config['context_window'])
                _, loss = self.model(xb, yb)
                losses.append(loss.item())
            out[split] = np.mean(losses)
        self.model.train()
        return out
