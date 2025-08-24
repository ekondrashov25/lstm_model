import numpy as np
import random

def save_checkpoint(model, filename):
    data = {k: v for k, v in model.get_params().items()}
    data.update({f"mem_{k}": v for k, v in model.mem.items()})
    np.savez_compressed(filename, **data)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, filename):
    checkpoint = np.load(filename)
    params = model.get_params()

    for k in params:
        if k in checkpoint:
            params[k][...] = checkpoint[k]
        else:
            print(f"Warning: parameter {k} not found in checkpoint")
            
    for k in model.mem:
        mem_key = f"mem_{k}"
        if mem_key in checkpoint:
            model.mem[k][...] = checkpoint[mem_key]
        else:
            print(f"Warning: memory {mem_key} not found in checkpoint")

    print(f"Loaded checkpoint from {filename}")

def calculate_perplexity(text, char_to_idx):
    if len(text) < 2:
        return float('inf')
    
    # Simple character-level perplexity
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    total_chars = len(text)
    log_prob = 0
    for char, count in char_counts.items():
        prob = count / total_chars
        log_prob += count * np.log(prob)
    
    return np.exp(-log_prob / total_chars)


def compute_validation_loss(model, data, char_to_idx, model_seq_length):
    start = random.randint(0, len(data) - model_seq_length - 1)
    inputs = [char_to_idx[ch] for ch in data[start:start+model_seq_length]]
    targets = [char_to_idx[ch] for ch in data[start+1:start+model_seq_length+1]]
    h_prev = np.zeros((model.hidden_size, 1))
    c_prev = np.zeros((model.hidden_size, 1))
    loss, _ = model.forward(inputs, targets, h_prev, c_prev)
    return float(loss) / model_seq_length