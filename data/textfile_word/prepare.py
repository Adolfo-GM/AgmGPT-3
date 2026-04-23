"""
Prepare the text file(s) for word-level language modeling.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""

import os
import pickle
import numpy as np
import re

base_dir = os.path.dirname(__file__)
data_chunks = []

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data_chunks.append(f.read())
            except:
                pass

data = ''.join(data_chunks)

if not data:
    raise ValueError("No .txt data found")
tokens = re.findall(r"[\w']+|[^\w\s]", data)
print(f"length of dataset in words: {len(tokens):,}")

import string
words = sorted(list(set(tokens)))
# Add all single characters to ensure immunity to OOV errors
all_chars = list(string.ascii_letters + string.digits + string.punctuation + " \n\r\t")
words = sorted(list(set(words + all_chars)))
vocab_size = len(words)
print("all the unique words:", words)
print(f"vocab size: {vocab_size:,}")

stoi = {w:i for i,w in enumerate(words)}
itos = {i:w for i,w in enumerate(words)}
def encode(seq):
    return [stoi[w] for w in seq]
def decode(l):
    return ' '.join([itos[i] for i in l])

n = len(tokens)
train_data = tokens[:int(n*0.9)]
val_data = tokens[int(n*0.9):]

train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(base_dir, 'train.bin'))
val_ids.tofile(os.path.join(base_dir, 'val.bin'))

meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
with open(os.path.join(base_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
