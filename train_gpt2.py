import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import funcitonal as F

class CasualSelfAttention(nn.Module):
    # all heads grouped together to run in parallel
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # k,q,v projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        # made it the have same dims as att for the masked_fill

    def forward(self, x):
        B, T, C = x.shape # batch size, sequence length, num embd (hs * nh)
        # nh = num heads, hs = head size
        qkv = self.c_attn(x) # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2) # (B, T, C)
        # make nh into a batch dimension so operations can be applied in parallel
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # multiply and scale by factor of sqrt(hs)
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf")) # mask future tokens
        att = F.softmax(att, dim=-1) # make attention sum to one
        y = att @ v # the weighted sum. (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        # transpose makes it not contiguous; we need contiguous for view()
        y = self.c_proj(y)
        return y



class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x) # linear expansion
        x = self.gelu(x) # gelu is relu but more smooth, so no dead relu neuron
        x = self.c_proj(x) # linear projection
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # residual connections
        x = x + self.attn(self.ln_1(x)) # communicate
        x = x + self.mlp(self.ln_2(x)) # think individually abt info gathered


@dataclass # automatically make init
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50527
    n_layer: int = 12
    n_head: int = 12
    n_embd : int = 768

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

    