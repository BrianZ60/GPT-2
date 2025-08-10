import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

class CasualSelfAttention(nn.Module):
    # all heads grouped together to run in parallel
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # k,q,v projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1.0

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
        self.c_proj.SCALE_INIT = 1.0 

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

        return x


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
            wte =  nn.Embedding(config.vocab_size, config.n_embd),
            wpe =  nn.Embedding(config.block_size, config.n_embd),
            h =    nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        # we do this because they are to an extent, inverses
        # we also expect the wte to react similarly for synonyms, and the lm_head to give synonyms similar scores
        # for more information, see https://arxiv.org/pdf/1608.05859 
        self.transformer.wte.weight = self.lm_head.weight # also saves a lot of parameters

        # init params
        self.apply(self._init_weights) # apply to every submodule

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # scale down weights of c_proj in mlp and attn
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        # idx: (B, T)
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # flatten logits into (B*T, vocab_size) and targets into (B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-x1"}
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {model_type}")

        # make config_args dict based on model_type
        config_args = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        # add two more args:
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        # unpack dict into args
        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")] # we don't want the mask buffer

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        # sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        # sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}" 

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T)

            else:
                assert sd_hf[k].shape == sd[k].shape # ,  f"Shape mismatch at key: {k}. {sd_hf[k].shape} != {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

device = "cuda" if torch.cuda.is_available() else "cpu"

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # load and store tokens
        enc = tiktoken.get_encoding("gpt2")
        with open("input.txt", "r") as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = 0
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T) 

        self.current_position += B*T

        # if next batch is out of bounds, loop back around
        if self.current_position + B*T+1 >= len(self.tokens):
            self.current_position = 0

        return x, y
    


train_loader = DataLoaderLite(B=4, T=32)


model = GPT(GPTConfig()).to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Step: {i}, loss: {loss.item()}")

import sys; sys.exit(0) # skip eval for now 



model.eval()
num_return_sequences = 5
max_length = 30

tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(dim=0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)


torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.inference_mode():
        logits = model(x) # (B, T, vocab_size)
        logits = logits[:, -1, :] # (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # (B, 50)
        # select a token
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        xcol = torch.gather(topk_indices, -1, ix) # get the element at the index for each batch
        x = torch.cat((x, xcol), dim=1)
for i in range(num_return_sequences):
    tokens = x[i].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)