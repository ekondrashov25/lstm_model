import numpy as np

from model import LSTMModel


filename = "datasets/code/oop_dataset_1000_lines.txt"
data = open(filename, 'r').read()
chars = sorted(list(set(data)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# hyperparameters of the model
# hidden_size=250
# learning_rate=1e-1
# seq_length=25

model = LSTMModel(input_size=len(chars), hidden_size=250, output_size=len(chars))
n, p = 0, 0
h_prev = np.zeros((model.hidden_size, 1))
c_prev = np.zeros((model.hidden_size, 1))
smooth_loss = -np.log(1.0 / len(chars)) * model.seq_length
sample_size = 500
epochs = 5
iterations = len(data) // model.seq_length * epochs
print(iterations)

for _ in range(iterations):
    if p + model.seq_length + 1 >= len(data) or n == 0:
        # reset memory
        h_prev = np.zeros((model.hidden_size, 1))
        c_prev = np.zeros((model.hidden_size, 1))
        p = 0

    # Prepare inputs and targets
    inputs = [char_to_idx[ch] for ch in data[p:p+model.seq_length]]
    targets = [char_to_idx[ch] for ch in data[p+1:p+model.seq_length+1]]

    # Forward and backward pass
    loss, cache = model.forward(inputs, targets, h_prev, c_prev)
    grads = model.backward(cache, inputs, targets)
    model.update_params(grads)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    sample_size = 500

    # Sample and print every 100 iterations
    if n % 100 == 0:
        grad_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
        print(f"iter {n}, loss: {smooth_loss:.4f}, grad norm: {grad_norm:.4f}, loss per char: {loss / model.seq_length:.4f}")
        if smooth_loss < 35:
            sample_ix = model.sample(inputs[0], sample_size, h_prev, c_prev)
            sample_text = ''.join(idx_to_char[ix] for ix in sample_ix)
            print(f"\n{sample_text}\n")
        # sample_ix = model.sample(inputs[0], 100, h_prev, c_prev)
        # sample_text = ''.join(idx_to_char[ix] for ix in sample_ix)
        # print("----%s----" % sample_text)

    # Update states and pointer
    h_prev = cache[1][model.seq_length-1]
    c_prev = cache[2][model.seq_length-1]
    p += model.seq_length
    n += 1