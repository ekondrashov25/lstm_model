import numpy as np


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