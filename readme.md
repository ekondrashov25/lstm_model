# LSTM Model Training and Process Mining

This repository contains an implementation of a Long Short-Term Memory (LSTM) neural network for sequence modeling tasks. The model is designed for educational and experimental purposes and assumed to use in proccess mining tasks

## XES Logger and Training Process Logging

During training, a custom XES logger (`XESLogger`) is used to record high-level events and statistics about the training process in the [XES (eXtensible Event Stream)](https://www.xes-standard.org/) format.

### What is logged?
- **Each epoch** is logged as a separate trace in the XES file.
- **Events** are logged at key points during training, including:
  - Forward and backward passes
  - Weights updates
  - Gradient clipping
  - Gate summary statistics
  - Sampling from the model
  - Learning rate decay
  - Validation loss checks
  - Plateau detection (when validation loss stops improving)
  - Gradient explosion detection
  - Early stopping (when training would stop due to lack of improvement)
  - End of epoch summary

### Structure of the `.xes` file
- **Trace**: Each trace corresponds to one epoch of training.
  - Trace attributes: `epoch`, `iterations` (number of iterations in the epoch)
- **Events**: Each event has a type (`concept:name`) and a set of attributes. Example event types and attributes:

| Event Type           | Key Attributes                                      | Description                                  |
|----------------------|-----------------------------------------------------|----------------------------------------------|
| ForwardPass          | iteration, loss, grad_norm, learning_rate           | Forward pass through the model               |
| BackwardPass         | iteration, grad_norm                                | Backward pass (gradient computation)         |
| WeightsUpdate        | iteration, weight_change_norm, update_category      | Model weights updated                        |
| GradientClipping     | iteration, grad_norm, threshold                     | Gradients clipped due to large norm          |
| GateSummary          | iteration, forget_mean, input_mean, output_mean     | Summary stats for LSTM gates                 |
| Sampling             | iteration, temperature, sample_text                 | Model generates a sample                     |
| LearningRateDecay    | iteration, old_lr, new_lr, reason                   | Learning rate reduced                        |
| Validation           | iteration, validation_loss                          | Validation loss computed                     |
| EarlyStoppingCheck   | iteration, triggered, patience_counter, best_val_loss| Check for early stopping                     |
| PlateauDetected      | iteration, patience_counter, best_val_loss, current_val_loss | Plateau in validation loss detected |
| GradientExplosion    | iteration, grad_norm, threshold, loss               | Gradient norm exceeded threshold             |
| EarlyStopping        | iteration, reason, best_val_loss                    | Early stopping triggered                     |
| EpochEnd             | final_loss, final_grad_norm                         | End of epoch summary                         |


### Literature
- Model architecture is written using the following [article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) and this [article](https://karpathy.github.io/2015/05/21/rnn-effectiveness/).
- Some pieces of code taken from this [repo](https://github.com/karpathy/char-rnn/blob/master/model/LSTM.lua)