import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import random

from models.lstm_model import LSTMModel
from utils import save_checkpoint
from scripts.xeslog import XESLogger

# run file:
# python training.py --dataset datasets/literature/shakespear.txt --hidden_size 250 --epochs 25 --sample_size 500

log = XESLogger()

# fet up logging to both file and console
log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler()
    ]
)

# for plotting graphs after training
loss_history = []
loss_per_char_history = []
iteration_history = []

def compute_validation_loss(model, data, char_to_idx, model_seq_length):
    start = random.randint(0, len(data) - model_seq_length - 1)
    inputs = [char_to_idx[ch] for ch in data[start:start+model_seq_length]]
    targets = [char_to_idx[ch] for ch in data[start+1:start+model_seq_length+1]]
    h_prev = np.zeros((model.hidden_size, 1))
    c_prev = np.zeros((model.hidden_size, 1))
    loss, _ = model.forward(inputs, targets, h_prev, c_prev)
    return float(loss) / model_seq_length

def train(args):
    filename = args.dataset
    data = open(filename, 'r').read()
    chars = sorted(list(set(data)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}

    hidden_size = args.hidden_size
    learning_rate = 1e-1
    sample_size = args.sample_size
    epochs = args.epochs
    patience = 3  # for early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    GRAD_EXPLOSION_THRESHOLD = 50.0  # adjust this value
    plateau_patience = 2  # number of validations with no improvement to consider as plateau
    plateau_logged = False  # track if plateau event has been logged for current plateau


    model = LSTMModel(input_size=len(chars), hidden_size=hidden_size, output_size=len(chars), learning_rate=learning_rate)

    n, p = 0, 0
    h_prev = np.zeros((model.hidden_size, 1))
    c_prev = np.zeros((model.hidden_size, 1))
    smooth_loss = -np.log(1.0 / len(chars)) * model.seq_length

    iterations_per_epoch = len(data) // model.seq_length
    total_iterations = iterations_per_epoch * epochs
    current_threshold = 3

    logging.info(f"epoch in training: {epochs}, iterations to complete: {total_iterations}, iterations per epoch={iterations_per_epoch}")
    logging.info(f"model parameters: {sample_size=}, input_size={len(chars)}, output_size={len(chars)}, {hidden_size=}, {learning_rate=}, {model.seq_length=}")

    for epoch in range(epochs):
        trace = log.add_trace(f"epoch_{epoch}")
        log.add_trace_attribute(trace, "epoch", epoch)
        log.add_trace_attribute(trace, "iterations", iterations_per_epoch)

        Wf_prev = model.W_f.copy()

        for i in range(iterations_per_epoch):
            if p + model.seq_length + 1 >= len(data) or n == 0:
                h_prev = np.zeros((model.hidden_size, 1))
                c_prev = np.zeros((model.hidden_size, 1))
                p = 0

            inputs = [char_to_idx[ch] for ch in data[p:p+model.seq_length]]
            targets = [char_to_idx[ch] for ch in data[p+1:p+model.seq_length+1]]

            loss, cache = model.forward(inputs, targets, h_prev, c_prev)
            grads = model.backward(cache, inputs, targets)
            grad_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
            model.update_params(grads)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            if i % 50 == 0: # forward pass
                log.add_event(trace, "ForwardPass", attrs={
                    "iteration": i,
                    "loss": float(smooth_loss),
                    "grad_norm": float(grad_norm),
                    "learning_rate": learning_rate
                })

            if i % 50 == 0: # backward pass
                log.add_event(trace, "BackwardPass", attrs={
                    "iteration": i,
                    "grad_norm": float(grad_norm)
                })

            if i % 100 == 0: # logging + weights update
                logging.info(f"iter {n}, loss: {smooth_loss:.4f}, grad norm: {grad_norm:.4f}, loss per char: {loss / model.seq_length:.4f}, current threshold: {current_threshold}")

                weight_change = np.linalg.norm(model.W_f - Wf_prev)
                update_category = "large" if weight_change > 1.0 else "small"
                log.add_event(trace, "WeightsUpdate", attrs={
                    "iteration": i,
                    "weight_change_norm": float(weight_change),
                    "update_category": update_category
                })
                Wf_prev = model.W_f.copy()

            if grad_norm > 5.0: # clipping gradients
                log.add_event(trace, "GradientClipping", attrs={
                    "iteration": i,
                    "grad_norm": float(grad_norm),
                    "threshold": 5.0
                })

            if i % 200 == 0: # gate summary
                xs, hs, cs, ps, is_, fs, os, gs = cache
                log.add_event(trace, "GateSummary", attrs={
                    "iteration": i,
                    "forget_mean": float(np.mean([fs[t] for t in fs])),
                    "input_mean": float(np.mean([is_[t] for t in is_])),
                    "output_mean": float(np.mean([os[t] for t in os])),
                    "forget_var": float(np.var([fs[t] for t in fs])),
                    "input_var": float(np.var([is_[t] for t in is_])),
                    "output_var": float(np.var([os[t] for t in os]))
                })

            if smooth_loss < 2.5 and i % 500 == 0: # sampling from model due to low loss
                sample_ix = model.sample(inputs[0], sample_size, h_prev, c_prev, temperature=0.8)
                sample_text = ''.join(idx_to_char[ix] for ix in sample_ix)
                log.add_event(trace, "Sampling", attrs={
                    "iteration": i,
                    "temperature": 0.8,
                    "sample_text": sample_text[:100]
                })

            if i > 0 and i % 2000 == 0: # learning rate decay, used to decrease lr
                old_lr = learning_rate
                learning_rate *= 0.5
                log.add_event(trace, "LearningRateDecay", attrs={
                    "iteration": i,
                    "old_lr": old_lr,
                    "new_lr": learning_rate,
                    "reason": "scheduled"
                })

            if i % 1000 == 0: 
                val_loss = compute_validation_loss(model, data, char_to_idx, model.seq_length) # like simulation of validation of model, it tooks random part of dataset and compute loss of model on it
                log.add_event(trace, "Validation", attrs={
                    "iteration": i,
                    "validation_loss": val_loss
                })

                if val_loss < best_val_loss: # early stopping check
                    best_val_loss = val_loss
                    patience_counter = 0
                    plateau_logged = False  # reset plateau flag on improvement
                else:
                    patience_counter += 1

                log.add_event(trace, "EarlyStoppingCheck", attrs={
                    "iteration": i,
                    "triggered": patience_counter >= patience,
                    "patience_counter": patience_counter,
                    "best_val_loss": best_val_loss
                })

                if patience_counter == plateau_patience and not plateau_logged: # plateau detection event
                    log.add_event(trace, "PlateauDetected", attrs={
                        "iteration": i,
                        "patience_counter": patience_counter,
                        "best_val_loss": best_val_loss,
                        "current_val_loss": val_loss
                    })
                    plateau_logged = True

                if grad_norm > GRAD_EXPLOSION_THRESHOLD: # gradient explosion
                    log.add_event(trace, "GradientExplosion", attrs={
                        "iteration": i,
                        "grad_norm": float(grad_norm),
                        "threshold": GRAD_EXPLOSION_THRESHOLD,
                        "loss": float(smooth_loss)
                    })
                    
                if patience_counter >= patience: # logging if early stopping
                    log.add_event(trace, "EarlyStopping", attrs={
                        "iteration": i,
                        "reason": "no improvement",
                        "best_val_loss": best_val_loss
                    })
                    
            h_prev = cache[1][model.seq_length - 1]
            c_prev = cache[2][model.seq_length - 1]
            p += model.seq_length
            n += 1

        log.add_event(trace, "т", attrs={
            "final_loss": float(smooth_loss),
            "final_grad_norm": float(grad_norm)
        })
       

    # save_checkpoint(model, 'checkpoints/model.npz')
    log.save_xes_file("lstm_log.xes")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='datasets/literature/shakespear.txt')
    parser.add_argument('--hidden_size', type=int, default=250)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--sample_size', type=int, default=500)
    args = parser.parse_args()
    train(args)
