import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

import time
import os
from datasets import load_dataset
from goldenswag import render_example
import random




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
        # att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf")) # mask future tokens
        # att = F.softmax(att, dim=-1) # make attention sum to one
        # y = att @ v # the weighted sum. (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        # Flash attention uses kernel fusion and avoids large reads/writes by using GPU on-chip memory more
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257
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



    def configure_optimizers(self, weight_decay, learning_rate, device):

        param_dict = {name : param for name, param in self.named_parameters() if param.requires_grad}

        # weight decay discourages model from relying too heavy on a weight by penalizing large weights
        # (we add a penalty to the loss that increases as weights get bigger)

        # we weight decay parameters that are 2D, like weight matrcies in linear layers and embeddings
        # biases and layernorms are not weight decayed
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params) # counts total number of elements (parameters)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)

        print("Configured optimizer with weight decay")
        print(f"- num decayed param tensors: {len(decay_params)}, totaling {num_decay_params:,} parameters")
        print(f"- num non-decayed parameter tensors: {len(no_decay_params)}, totaling {num_no_decay_params:,} parameters")

        use_fused = "cuda" in device
        print(f"- fused AdamW: {use_fused}\n")

        optimizer = torch.optim.AdamW(params=optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer




import tiktoken
import numpy as np

def load_tokens(filename):
    """
    Load the tokens from a shard file and return them as a tensor
    """
    np_tokens = np.load(filename)
    pytorch_tokens = torch.tensor(np_tokens, dtype=torch.long)
    return pytorch_tokens

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        # get each shard's filename
        data_root_dir = "fineweb-edu-10B"
        shards = os.listdir(data_root_dir) # get all file names
        shards = [shard_name for shard_name in shards if split in shard_name]
        shards = sorted(shards)
        shards = [os.path.join(data_root_dir, shard_name) for shard_name in shards]

        self.shards = shards

        assert len(shards) > 0, f"No shards found for split {split}!"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.shard_tokens = load_tokens(filename=self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank


    def next_batch(self):
        B, T = self.B, self.T
        buf = self.shard_tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T * self.num_processes # this way, we train on a block of num_processes batches in parallel

        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + B * T * self.num_processes + 1 >= len(self.shard_tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards) # loop shard if need to
            self.shard_tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


def get_most_likely_row(tokens, mask, logits):
    
    shift_logits = (logits[:, :-1, :]).contiguous() # we don't want the logits for the next token after the ending
    shift_tokens = (tokens[:, 1:]).contiguous() # get all the tokens except for the first; the next tokens

    # flatten to calculate CE loss
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)

    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none") # default reduction is "mean", but we want a tensor of losses for each token

    shift_losses = shift_losses.view(tokens.size(0), -1)

    shift_mask = (mask[:, 1:]).contiguous() 
    masked_shift_losses = shift_losses * shift_mask # only want losses of ending tokens

    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1) # avg loss for each completion

    pred = avg_loss.argmin().item()

    return pred


def load_model_from_save(model, save_path, optimizer, device):
    save = torch.load(save_path, map_location=device)

    model.load_state_dict(save["model"])

    start_step = save["step"] + 1
    val_save_loss = save["val_loss"]

    optimizer.load_state_dict(save["optimizer"])

    torch.set_rng_state(save["rng_state"])
    torch.cuda.set_rng_state_all(save["cuda_rng_state"])

    print(f"Loaded model from train step {start_step} with val loss {val_save_loss}")

    return start_step, val_save_loss



# ------------------------------------ TRAIN MODEL ---------------------------------------------




# When using DDP, we don't just do python train_gpt2.py, instead, we do
# torchrun --standalone --nproc_per_node=num_gpus train_gpt2.py
# torchrun would set the RANK, LOCAL_RANK, and WORLD_SIZE env variables

if __name__ == "__main__":
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    # we will use DDP (distributed data parallel to distribute processes among the gpus)
    import torch.distributed as dist


    ddp = int(os.environ.get("RANK", -1)) != -1 # checks if RANK var exists in environment

    if ddp:
        assert torch.cuda.is_available(), "lets use CUDA please"
        init_process_group(backend="nccl") # nccl for CUDA
        ddp_rank = int(os.environ["RANK"]) # global id for each process (same as local b/c only one machine)
        ddp_local_rank = int(os.environ["LOCAL_RANK"]) # local id for each process
        ddp_world_size = int(os.environ["WORLD_SIZE"]) # total num of processes (GPUs)

        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # master process will do logging, checkpointing
    else:
        # no DDP
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")


    device_type = "cuda" if device.startswith("cuda") else "cpu"


    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    enc = tiktoken.get_encoding("gpt2")

    # print("NOTE: Remove 1/32 scale factor when on cloud gpu")
    total_batch_size = 524288 # measured in total number of tokens
    # We use this b/c it is 2**19, close to openai's 0.5M

    # We simulate 0.5M batch size by doing many forward, backward passes, accumulating the gradient

    B = 4 # 128 # micro batch size
    T = 1024 # sequence length (num tokens)

    assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # each process is doing B*T tokens each micro-step
    grad_accum_steps = 1
    if master_process:
        print(f"desired total batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

    # print("NOTE: Use TF32 when on cloud GPU")
    torch.set_float32_matmul_precision("high") # use TF32 (lower precision then FP32, but faster)
    # variables are still FP32, but the matrix multi are TF32


    model = GPT(GPTConfig(vocab_size=50304)).to(device) # use 50304 instead of 50257 b/c it is a much nicer number (divisible by 128)
    compiled_model = torch.compile(model) if device_type == "cuda" else model
    # What torch.compile() does:
    # 1. Views the entire network as a whole, allowing for more efficient processing and minimizes Python interpreter overhead
    # 2. Reduces read/write time btwn gpu and memory with operator fusion. This also mitigates memory bandwidth cost.

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank]) # synchronize gradients across processes
        # every process will have the avg of the gradients across all the processes

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715 # openai warmed up for 375M tokens, 375M // total_batch_size = 715
    decay_steps = 19073 
    max_steps = 19073 # 10B tokens // total_batch_size = steps for one epoch


    def get_lr(it):
        # linear warmup for first warmup_steps steps
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps # linearly increasing lr to max_Lr

        # use min_lr after we do our lr decay
        if it > decay_steps:
            return min_lr

        # when in between, we cosine decay to min lr
        decay_ratio = (it - warmup_steps) / (decay_steps - warmup_steps) # don't include the warmup steps
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # start at 1 and goes to 0
        
        return min_lr + (max_lr - min_lr) * coeff


    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

    log_dir = "log"
    os.makedirs(log_dir, exist_ok = True)
    log_file = os.path.join(log_dir, "log.txt")

    # with open(log_file, "w") as f:
    #     pass # clear the log_file

    # save_model(model, save_path="log/model_19072.pt", device=device)



    prompt = "Data visualization empowers users to "

    model.generate_text(model, num_return_sequences = 4, max_new_tokens=100, prompt=prompt, temp=0.8)



    for step in range(max_steps):
       
        t0 = time.time()

        last_step = (step == max_steps - 1)

        # evaluate using val data every 250 steps
        if step % 250 == 0 or last_step:
        # if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.inference_mode():
                val_loss_accum = 0.0
                val_loss_steps = 1 # 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    if device_type == "cuda": 
                        with torch.autocast(device_type=device, dtype=torch.bfloat16):
                            logits, loss = compiled_model(x, y)
                    else: logits, loss = model(x, y)
                    loss /= val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"Validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f: # a is for append
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                
                if step > 0 and (step % 500 == 0 or last_step):
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")

                    checkpoint = {
                        "model": model.state_dict(),
                        "step": step,
                        "val_loss": val_loss_accum.item(),
                        "optimizer": optimizer.state_dict(),
                        "rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state_all(),
                    }

                    torch.save(checkpoint, checkpoint_path)
                    print(f"Checkpoint saved to {checkpoint_path}")

        # evaluate using goldenswag every 250 steps
        if master_process and (step % 250 == 0 or last_step):
            model.eval()

            num_correct = 0
            num_examples = 0

            gs = load_dataset("PleIAs/GoldenSwag", split="validation")
            for i, example in enumerate(gs):

                tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)

                with torch.inference_mode():
                    if device_type == "cuda":
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            logits, _ = model(tokens)
                    else:
                        logits, _ = model(tokens)

                    pred = get_most_likely_row(tokens, mask, logits)

                num_examples += 1
                num_correct += int(pred == int(label))

                if num_examples == 5:
                    break


                    
                

            if ddp:
                # sync across all processes
                num_examples = torch.tensor(num_examples, dtype=torch.long, device=device)
                num_correct = torch.tensor(num_correct, dtype=torch.long, device=device)
                
                dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct, op=dist.ReduceOp.SUM)

                num_examples = num_examples.item()
                num_correct = num_correct.item()

            acc = num_correct / num_examples
            
            print(f"Goldenswag accuracy: {num_correct}/{num_examples}={acc:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} goldenswag {acc:.4f}\n")

            



        # generate samples every 250 steps (not including step 0)
        if step > 0 and step % 250 == 0 or last_step:
            sample_rng = torch.Generator(device=device)

            model.generate_text(4, 32, "The study of quantum mechanics", sample_rng)
            
            

        model.train()
        
        optimizer.zero_grad() # only zero grad every grad_accum_steps

        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # if micro_step==0 and step==0: print("NOTE: Enable autocast for bf16 when on cloud gpu")
            if device_type == "cuda":
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = compiled_model(x, y) # actually changes the datatype of logits, but others remain FP32 (mixed precision)
            else:
                logits, loss = model(x, y)
            loss /= grad_accum_steps
            loss_accum += loss.detach() # don't want to track gradients here
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # don't need to grad sync every micro step
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # averages the loss_accum across all processes and deposits that value

        # square all the gradients, add them up, and take sqrt to get grad norm
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # cap gradient norm at 1.0

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        
        if device_type == "cuda":
            torch.cuda.synchronize() # wait for gpu to finish scheduled work before continuing
        t1 = time.time()
        dt = t1 - t0 # time diff in seconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt

        if master_process:
            print(f"step: {step:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms, tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    if ddp:
        destroy_process_group()