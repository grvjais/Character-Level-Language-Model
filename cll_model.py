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
    
