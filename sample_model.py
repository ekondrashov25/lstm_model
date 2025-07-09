import numpy as np

from utils import load_checkpoint
from models.lstm_model import LSTMModel

hidden_size = 250 # should match hidden_size which was set during traninig process
checkpoint_file = "checkpoints/model.npz"

filename = "datasets/literature/shakespear.txt"
data = open(filename, 'r').read()
chars = sorted(list(set(data)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()} 

model = LSTMModel(input_size=len(chars), hidden_size=hidden_size, output_size=len(chars))
load_checkpoint(model, checkpoint_file)

h = np.zeros((model.hidden_size, 1))
c = np.zeros((model.hidden_size, 1))

seed_char = 'T'
seed_idx = char_to_idx.get(seed_char, 0)

sample_ix = model.sample(seed_idx, 500, h, c)
sample_text = ''.join(idx_to_char[ix] for ix in sample_ix)

print(f"\nGenerated text:\n{sample_text}\n")