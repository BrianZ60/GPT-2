"""
Loads and evaluates GoldenSwag
GoldenSwag is a subset of HellaSwag that was heavily filtered in hopes
of ensuring that the benchmark genuinely requires commonsense reasoning.
For more information: https://arxiv.org/pdf/2504.07825

The validation set has a total of 1525 examples, or ~15.2% of the original dataset.
"""

import os
from datasets import load_dataset
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

enc = tiktoken.get_encoding("gpt2")

def render_example(example):
    """
    Render a GoldenSwag multiple-choice example into tensors for model evaluation.

    Args:
        example (dict): A single GoldenSwag example with keys:
            - "ctx": A string containing the shared context.
            - "endings": A list of four string completions that serve as the choices.
            - "label": An integer (0-3) indicating the index of the correct ending.

    Returns:
        tokens: A tensor with shape (4, N) containing the tokens of the context + each ending.
        mask: A tensor the same shape as tokens with values
            - 0 for context tokens
            - 1 for ending tokens (where we will evaluate probabilities).
        label: The index of the correct ending as provided in the example.
    """

    ctx = example["ctx"]
    endings = example["endings"]
    label = example["label"]

    ctx_tokens = enc.encode(ctx)
    tok_rows = []
    mask_rows = []

    for ending in endings:
        end_tokens = enc.encode(" " + ending)  # GPT-2 tokenizer needs prepending " "
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens)) # 0 for context tokens, 1 for ending tokens
    
    # get the maximum row length and make it the sequence length (T) 
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return tokens, mask, label


@torch.inference_mode()
def evaluate(model_type, device):
    torch.set_float32_matmul_precision("high") # tf32
    model = GPT2LMHeadModel.from_pretrained(model_type).to(device)
    # model = torch.compile(model)


    num_correct = 0
    num_examples = 0

    gs = load_dataset("PleIAs/GoldenSwag", split="validation")
    for i, example in enumerate(gs):
        tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        logits = model(tokens).logits

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

        num_examples += 1
        num_correct += int(pred == int(label))

        print(f"Evaluated {num_examples} examples | acc: {num_correct}/{num_examples}={num_correct/num_examples:.4f}")

        if num_examples < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) | {end}")
            print(f"predicted: {pred}, actual: {label}")
        
        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    args = parser.parse_args()
    evaluate(args.model_type, args.device)


    



