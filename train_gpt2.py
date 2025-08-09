from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import funcitonal as F


@dataclass # automatically make init
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd : int = 384

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # use module dict to replicate structure of the hf model
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    