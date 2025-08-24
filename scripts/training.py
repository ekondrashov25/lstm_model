import os
from datetime import datetime

import logging
import numpy as np

from models.lstm_model import LSTMModel
from scripts.xeslog import XESLogger
from utils import calculate_perplexity, compute_validation_loss
from scripts.detailed_xeslogger import DetailedEventLogger

# run file:
# python training.py --dataset datasets/literature/shakespear.txt --hidden_size 250 --epochs 25 --sample_size 500

log = XESLogger()

# initialize the detailed event logger
detailed_logger = DetailedEventLogger(log)

log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training.log')

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler()])


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

    # add dataset metadata logging
    log.add_event(log.add_trace("dataset_info"), "DatasetInfo", attrs={
        "dataset_name": filename.split('/')[-1],
        "dataset_size": len(data),
        "vocabulary_size": len(chars),
        "unique_chars": ''.join(chars[:50]) + "..." if len(chars) > 50 else ''.join(chars)
    })

    model = LSTMModel(input_size=len(chars), hidden_size=hidden_size, output_size=len(chars), learning_rate=learning_rate)

    # add model architecture logging
    log.add_event(log.add_trace("model_architecture"), "ModelArchitecture", attrs={
        "input_size": len(chars),
        "hidden_size": hidden_size,
        "output_size": len(chars),
        "learning_rate": learning_rate,
        "sequence_length": model.seq_length,
        "total_parameters": sum(p.size for p in model.get_params().values())
    })

    n, p = 0, 0
    h_prev = np.zeros((model.hidden_size, 1))
    c_prev = np.zeros((model.hidden_size, 1))
    smooth_loss = -np.log(1.0 / len(chars)) * model.seq_length

    iterations_per_epoch = len(data) // model.seq_length
    total_iterations = iterations_per_epoch * epochs

    logging.info(f"epoch in training: {epochs}, iterations to complete: {total_iterations}, iterations per epoch={iterations_per_epoch}")
    logging.info(f"model parameters: {sample_size=}, input_size={len(chars)}, output_size={len(chars)}, {hidden_size=}, {learning_rate=}, {model.seq_length=}")

    for epoch in range(epochs):
        trace = log.add_trace(f"epoch_{epoch}")
        log.add_trace_attribute(trace, "epoch", epoch)
        log.add_trace_attribute(trace, "iterations", iterations_per_epoch)
        log.add_trace_attribute(trace, "dataset", filename.split('/')[-1])

        Wf_prev = model.W_f.copy()
        epoch_start_time = datetime.now()

        # reset epoch statistics
        epoch_losses = []
        epoch_grad_norms = []
        epoch_weight_changes = []

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

            # track statistics
            epoch_losses.append(float(loss))
            epoch_grad_norms.append(float(grad_norm))

            if i % 50 == 0: # forward pass and backward pass with detailed logging
                # log detailed ForwardPass events
                detailed_logger.log_forward_pass_details(trace, i, cache, attrs={
                    "loss": float(smooth_loss),
                    "grad_norm": float(grad_norm),
                    "learning_rate": learning_rate,
                    "sequence_position": p,
                    "data_coverage": (p / len(data)) * 100
                })

                # log detailed BackwardPass events
                detailed_logger.log_backward_pass_details(trace, i, grads, cache, attrs={
                    "gradient_sparsity": float(np.mean([np.mean(g == 0) for g in grads.values()]))
                })
                
            if i % 100 == 0: # logging + weights update with detailed logging
                logging.info(f"iter {n}, loss: {smooth_loss:.4f}, grad norm: {grad_norm:.4f}, loss per char: {loss / model.seq_length:.4f}")

                # log detailed weight update events
                detailed_logger.log_weight_update_details(trace, i, model, Wf_prev)
                
                weight_change = np.linalg.norm(model.W_f - Wf_prev)
                update_category = "large" if weight_change > 1.0 else "small"
                epoch_weight_changes.append(float(weight_change))
                
                log.add_event(trace, "WeightsUpdate", attrs={
                    "iteration": i,
                    "weight_change_norm": float(weight_change),
                    "update_category": update_category,
                    "parameter_norm": float(np.linalg.norm(model.W_f)),
                    "has_weight_details": 1
                })
                Wf_prev = model.W_f.copy()

            if grad_norm > 5.0: # clipping gradients
                log.add_event(trace, "GradientClipping", attrs={
                    "iteration": i,
                    "grad_norm": float(grad_norm),
                    "threshold": 5.0,
                    "clipping_ratio": float(grad_norm / 5.0)
                })

            if i % 200 == 0: # gate summary
                xs, hs, cs, ps, is_, fs, os, gs = cache
                
                # calculate gate statistics
                forget_means = [float(np.mean(fs[t])) for t in fs]
                input_means = [float(np.mean(is_[t])) for t in is_]
                output_means = [float(np.mean(os[t])) for t in os]
                
                log.add_event(trace, "GateSummary", attrs={
                    "iteration": i,
                    "forget_mean": float(np.mean(forget_means)),
                    "input_mean": float(np.mean(input_means)),
                    "output_mean": float(np.mean(output_means)),
                    "forget_var": float(np.var(forget_means)),
                    "input_var": float(np.var(input_means)),
                    "output_var": float(np.var(output_means)),
                    "gate_correlation": float(np.corrcoef(forget_means, input_means)[0,1]) if len(forget_means) > 1 else 0.0
                })

            if smooth_loss < 2.5 and i % 500 == 0: # sampling from model due to low loss
                sample_ix = model.sample(inputs[0], sample_size, h_prev, c_prev, temperature=0.8)
                sample_text = ''.join(idx_to_char[ix] for ix in sample_ix)
                
                # calculate sample quality metrics
                sample_perplexity = calculate_perplexity(sample_text, char_to_idx)
                
                log.add_event(trace, "Sampling", attrs={
                    "iteration": i,
                    "temperature": 0.8,
                    "sample_text": sample_text[:100],
                    "sample_perplexity": sample_perplexity,
                    "unique_chars_in_sample": len(set(sample_text))
                })

            if i > 0 and i % 2000 == 0: # learning rate decay
                old_lr = learning_rate
                learning_rate *= 0.5
                log.add_event(trace, "LearningRateDecay", attrs={
                    "iteration": i,
                    "old_lr": old_lr,
                    "new_lr": learning_rate,
                    "reason": "scheduled",
                    "decay_factor": 0.5
                })

            if i % 1000 == 0: 
                val_loss = compute_validation_loss(model, data, char_to_idx, model.seq_length) # like simulation of validation of model, it tooks random part of dataset and compute loss of model on it
                log.add_event(trace, "Validation", attrs={
                    "iteration": i,
                    "validation_loss": val_loss,
                    "train_val_ratio": float(smooth_loss / val_loss) if val_loss > 0 else 0.0
                })

                if val_loss < best_val_loss: # early stopping check
                    best_val_loss = val_loss
                    patience_counter = 0
                    plateau_logged = False  # reset plateau flag on improvement
                else:
                    patience_counter += 1

                log.add_event(trace, "EarlyStoppingCheck", attrs={
                    "iteration": i,
                    "triggered": 1 if patience_counter >= patience else 0,
                    "patience_counter": patience_counter,
                    "best_val_loss": best_val_loss,
                    "improvement": float(best_val_loss - val_loss) if val_loss < best_val_loss else 0.0
                })

                if patience_counter == plateau_patience and not plateau_logged: # plateau detection event
                    log.add_event(trace, "PlateauDetected", attrs={
                        "iteration": i,
                        "patience_counter": patience_counter,
                        "best_val_loss": best_val_loss,
                        "current_val_loss": val_loss,
                        "plateau_duration": patience_counter
                    })
                    plateau_logged = True

                if grad_norm > GRAD_EXPLOSION_THRESHOLD: # gradient explosion
                    log.add_event(trace, "GradientExplosion", attrs={
                        "iteration": i,
                        "grad_norm": float(grad_norm),
                        "threshold": GRAD_EXPLOSION_THRESHOLD,
                        "loss": float(smooth_loss),
                        "explosion_factor": float(grad_norm / GRAD_EXPLOSION_THRESHOLD)
                    })
                    
                if patience_counter >= patience: # logging if early stopping
                    log.add_event(trace, "EarlyStopping", attrs={
                        "iteration": i,
                        "reason": "no improvement",
                        "best_val_loss": best_val_loss,
                        "epochs_without_improvement": patience_counter
                    })
                    
            h_prev = cache[1][model.seq_length - 1]
            c_prev = cache[2][model.seq_length - 1]
            p += model.seq_length
            n += 1

        # end of epoch summary with detailed statistics
        epoch_end_time = datetime.now()
        epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()
        
        log.add_event(trace, "EpochEnd", attrs={
            "final_loss": float(smooth_loss),
            "final_grad_norm": float(grad_norm),
            "epoch_duration_seconds": epoch_duration,
            "avg_loss": float(np.mean(epoch_losses)),
            "avg_grad_norm": float(np.mean(epoch_grad_norms)),
            "max_grad_norm": float(np.max(epoch_grad_norms)),
            "avg_weight_change": float(np.mean(epoch_weight_changes)),
            "iterations_per_second": iterations_per_epoch / epoch_duration
        })
       
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
