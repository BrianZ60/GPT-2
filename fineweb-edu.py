"""
Uses 10B token subset of FineWeb-Edu dataset
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data, and then saves the data shards to the "edu_fineweb10B" local directory
"""

import os
import multiprocessing as mp # running multiple processes in parallel
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "fineweb-edu-10B"

shard_size = int(1e8) # 100M tokens per shard, 100 total shards

# create the local dir for the data
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir) # __file__ is the path to this file, dirname gets the folder
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# fw_edu = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")
fw_edu = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")


enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]

def tokenize(doc):
    """tokenizes a single document from the dataset and returns a numpy array of uint16 tokens"""
    tokens = [eot] # end of text token seperates documents from each other
    tokens.extend(enc.encode_ordinary(doc["text"])) # ignore special tokens (like eot) in the dataset
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "document token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16) # save memory
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    """saves tokens numpy array to a binary file in .npy format"""
    np.save(filename, tokens_np)


if __name__ == "__main__":
    # tokenize all documents and write output shards
    num_processes = max(1, os.cpu_count() // 2)
    with mp.Pool(num_processes) as pool: # make pool with num_processes worker processes
        shard_idx = 0
        shard_tokens_np = np.empty((shard_size, ), dtype=np.uint16)
        shard_token_count = 0
        progress_bar = None
        
        # maps the tokenize function onto all items (documents) in fw_edu
        for tokens in pool.imap(tokenize, fw_edu, chunksize=16): # disbatch chunks of 16 at a time to each worker

            if shard_token_count + len(tokens) <= shard_size:
                shard_tokens_np[shard_token_count : shard_token_count + len(tokens)] = tokens
                shard_token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_idx}")
                    progress_bar.update(shard_token_count) # add in leftover tokens
                else:
                    progress_bar.update(len(tokens))

            else:
                # write the current shard to a file and start a new shard
                split = "val" if shard_idx == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"fineweb-edu_{split}_{shard_idx:06d}")

                # put whatever fits from the document into this shard, and put rest in next shard
                remainder = shard_size - shard_token_count
                progress_bar.update(remainder)

                shard_tokens_np[shard_token_count : shard_token_count + remainder] = tokens[:remainder]
                write_datafile(filename, shard_tokens_np)

                shard_idx += 1
                progress_bar = None

                # put the leftovers (len(tokens) - remainder) of the current document into the next shard
                leftover_tokens = len(tokens) - remainder

                shard_tokens_np[0 : leftover_tokens] = tokens[remainder:]
                shard_token_count = leftover_tokens


        # write any remaining tokens as last shard
        if shard_token_count != 0:
            split = "val" if shard_idx == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"fineweb-edu_{split}_{shard_idx:06d}")
            write_datafile(filename, shard_tokens_np[:shard_token_count])


