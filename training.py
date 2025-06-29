import numpy as np
import matplotlib.pyplot as plt


from model import LSTMModel
from utils import save_checkpoint


# hyperparameters of the model
# hidden_size=250
# learning_rate=1e-1
# seq_length=25

filename = "datasets/literature/shakespear.txt"
data = open(filename, 'r').read()
chars = sorted(list(set(data)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

hidden_size = 250
learning_rate = 1e-1

model = LSTMModel(input_size=len(chars), hidden_size=hidden_size, output_size=len(chars), learning_rate=learning_rate)

n, p = 0, 0
h_prev = np.zeros((model.hidden_size, 1))
c_prev = np.zeros((model.hidden_size, 1))
smooth_loss = -np.log(1.0 / len(chars)) * model.seq_length

sample_size = 500
epochs = 15
iterations = len(data) // model.seq_length * epochs

print(f"epoch in training: {epochs}, iterations to complete: {iterations}")
print(f"model parameters: {sample_size=}, input_size={len(chars)}, output_size={len(chars)}, {hidden_size=}, {learning_rate=}, {model.seq_length=}")

loss_history = []
loss_per_char_history = []
iteration_history = []

for _ in range(iterations):
    if p + model.seq_length + 1 >= len(data) or n == 0:
        # reset memory
        h_prev = np.zeros((model.hidden_size, 1))
        c_prev = np.zeros((model.hidden_size, 1))
        p = 0

    # prepare inputs and targets
    inputs = [char_to_idx[ch] for ch in data[p:p+model.seq_length]]
    targets = [char_to_idx[ch] for ch in data[p+1:p+model.seq_length+1]]

    # forward and backward pass
    loss, cache = model.forward(inputs, targets, h_prev, c_prev)
    grads = model.backward(cache, inputs, targets)
    model.update_params(grads)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    sample_size = 500

    if n % 100 == 0:
        # save_checkpoint(model, 'checkpoints/model.npz')
        grad_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
        print(f"iter {n}, loss: {smooth_loss:.4f}, grad norm: {grad_norm:.4f}, loss per char: {loss / model.seq_length:.4f}")

        loss_history.append(smooth_loss)
        loss_per_char_history.append(loss / model.seq_length)
        iteration_history.append(n)

        if smooth_loss < 20:
            sample_ix = model.sample(inputs[0], sample_size, h_prev, c_prev, temperature=0.8)
            sample_text = ''.join(idx_to_char[ix] for ix in sample_ix)
            print(f"\n{sample_text}\n")

    # update states and pointer
    h_prev = cache[1][model.seq_length-1]
    c_prev = cache[2][model.seq_length-1]
    p += model.seq_length
    n += 1

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(iteration_history, loss_history, label='Smooth loss')
plt.xlabel('Iteration')
plt.ylabel('Smooth loss')
plt.title('Smooth Loss over Iterations')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(iteration_history, loss_per_char_history, label='Loss per char', color='orange')
plt.xlabel('Iteration')
plt.ylabel('Loss per char')
plt.title('Loss per Character over Iterations')
plt.legend()

plt.tight_layout()
plt.show()