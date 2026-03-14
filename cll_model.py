import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

#------------------------------------------------------------

@dataclass

class ModelConfig:
    block_size: int = None
    vocab_size: int = None

    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4 

#---------------------------------------------------------------------

class Bigram(nn.Module):

    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros((n,n)))

    def get_block_size(self):
        return 1
    
    def forward(self, idx, targets=None):
        logits = self.logits[idx]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
    
# -----------------------------------------------------------------------------
# Helper functions for evaluating and sampling

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def print_samples(num=10):
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    top_k = args.top_k if args.top_k != -1 else None
    steps = train_dataset.get_output_length() - 1 
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')
    
    print('-'*80)
    for i in range(X_samp.size(0)):
        row = X_samp[i, 1:].tolist()
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        print(word_samp)
    print('-'*80)

@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train()
    return mean_loss

# -----------------------------------------------------------------------------
# Dataset logic

class CharDataset(Dataset):
    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}
        self.itos = {i:s for s,i in self.stoi.items()}

    def __len__(self):
        return len(self.words)

    def get_vocab_size(self):
        return len(self.chars) + 1 

    def get_output_length(self):
        return self.max_word_length + 1 

    def encode(self, word):
        return torch.tensor([self.stoi[w] for w in word], dtype=torch.long)

    def decode(self, ix):
        return ''.join(self.itos[i] for i in ix)

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 
        return x, y

def create_datasets(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()
    words = [w.strip() for w in data.splitlines() if w.strip()]
    chars = sorted(list(set(''.join(words))))
    max_word_length = max(len(w) for w in words)
    
    test_set_size = min(1000, int(len(words) * 0.1))
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]

    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)
    return train_dataset, test_dataset

class InfiniteDataLoader:
    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bigram MakeMore")
    parser.add_argument('--input-file', '-i', type=str, default='names.txt', help="input text file")
    parser.add_argument('--max-steps', type=int, default=5000, help="max training steps")
    parser.add_argument('--device', type=str, default='cpu', help="cpu or cuda")
    parser.add_argument('--batch-size', '-b', type=int, default=32)
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-1) # Higher LR for Bigram
    parser.add_argument('--top-k', type=int, default=-1)
    args = parser.parse_args()

    # Init dataset
    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    
    # Init Bigram Model
    config = ModelConfig(vocab_size=vocab_size)
    model = Bigram(config).to(args.device)
    
    # Init optimizer (Bigrams can handle much higher learning rates than Transformers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size)

    # Training Loop
    print("Starting training...")
    for step in range(args.max_steps):
        X, Y = [t.to(args.device) for t in batch_loader.next()]

        logits, loss = model(X, Y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")
            print_samples(num=5)

    print("Training complete! Final samples:")
    print_samples(num=10)