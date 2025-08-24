import numpy as np

class DetailedEventLogger:
    def __init__(self, xes_logger):
        self.log = xes_logger
        self.weight_matrices = ['W_f', 'W_i', 'W_c', 'W_o', 'W_hy']
        self.bias_vectors = ['b_f', 'b_i', 'b_c', 'b_o', 'b_y']
        self.gates = ['forget', 'input', 'output', 'candidate']
    
    def log_forward_pass_details(self, trace, iteration, cache, attrs=None):
        xs, hs, cs, ps, is_, fs, os, gs = cache
        
        main_attrs = attrs or {}
        main_attrs.update({
            "iteration": iteration,
            "sequence_length": len(xs),
            "has_gate_details": 1
        })
        
        self.log.add_event(trace, "ForwardPass", attrs=main_attrs)
        
        gate_stats = {
            'forget': fs,
            'input': is_,
            'output': os,
            'candidate': gs
        }
        
        for gate_name, gate_values in gate_stats.items():
            if gate_values:
                gate_means = [float(np.mean(gate_values[t])) for t in gate_values]
                gate_stds = [float(np.std(gate_values[t])) for t in gate_values]
                
                self.log.add_event(trace, f"Gate_{gate_name.capitalize()}", attrs={
                    "iteration": iteration,
                    "gate_type": gate_name,
                    "mean_activation": float(np.mean(gate_means)),
                    "std_activation": float(np.mean(gate_stds)),
                    "min_activation": float(np.min(gate_means)),
                    "max_activation": float(np.max(gate_means)),
                    "activation_range": float(np.max(gate_means) - np.min(gate_means)),
                    "parent_event": "ForwardPass"
                })
        
        if cs:
            cell_values = np.concatenate([cs[t] for t in cs], axis=1)
            self.log.add_event(trace, "CellState_Update", attrs={
                "iteration": iteration,
                "mean_value": float(np.mean(cell_values)),
                "std_value": float(np.std(cell_values)),
                "min_value": float(np.min(cell_values)),
                "max_value": float(np.max(cell_values)),
                "parent_event": "ForwardPass"
            })
        
        if hs:
            hidden_values = np.concatenate([hs[t] for t in hs], axis=1)
            self.log.add_event(trace, "HiddenState_Update", attrs={
                "iteration": iteration,
                "mean_value": float(np.mean(hidden_values)),
                "std_value": float(np.std(hidden_values)),
                "min_value": float(np.min(hidden_values)),
                "max_value": float(np.max(hidden_values)),
                "parent_event": "ForwardPass"
            })
    
    def log_backward_pass_details(self, trace, iteration, grads, cache, attrs=None):
        xs, hs, cs, ps, is_, fs, os, gs = cache
        
        main_attrs = attrs or {}
        main_attrs.update({
            "iteration": iteration,
            "gradient_norm": float(np.sqrt(sum(np.sum(g**2) for g in grads.values()))),
            "has_weight_details": 1
        })
        self.log.add_event(trace, "BackwardPass", attrs=main_attrs)
        
        for matrix_name in self.weight_matrices:
            if matrix_name in grads:
                grad_matrix = grads[matrix_name]
                grad_norm = float(np.linalg.norm(grad_matrix))
                grad_mean = float(np.mean(grad_matrix))
                grad_std = float(np.std(grad_matrix))
                
                self.log.add_event(trace, f"WeightGradient_{matrix_name}", attrs={
                    "iteration": iteration,
                    "matrix_name": matrix_name,
                    "gradient_norm": grad_norm,
                    "gradient_mean": grad_mean,
                    "gradient_std": grad_std,
                    "gradient_min": float(np.min(grad_matrix)),
                    "gradient_max": float(np.max(grad_matrix)),
                    "gradient_sparsity": float(np.mean(grad_matrix == 0)),
                    "parent_event": "BackwardPass"
                })
        
        for bias_name in self.bias_vectors:
            if bias_name in grads:
                grad_bias = grads[bias_name]
                grad_norm = float(np.linalg.norm(grad_bias))
                grad_mean = float(np.mean(grad_bias))
                grad_std = float(np.std(grad_bias))
                
                self.log.add_event(trace, f"BiasGradient_{bias_name}", attrs={
                    "iteration": iteration,
                    "bias_name": bias_name,
                    "gradient_norm": grad_norm,
                    "gradient_mean": grad_mean,
                    "gradient_std": grad_std,
                    "gradient_min": float(np.min(grad_bias)),
                    "gradient_max": float(np.max(grad_bias)),
                    "parent_event": "BackwardPass"
                })
        
        gate_gradients = {
            'forget': 'W_f',
            'input': 'W_i', 
            'output': 'W_o',
            'candidate': 'W_c'
        }
        
        for gate_name, matrix_name in gate_gradients.items():
            if matrix_name in grads:
                grad_matrix = grads[matrix_name]
                self.log.add_event(trace, f"GateGradient_{gate_name.capitalize()}", attrs={
                    "iteration": iteration,
                    "gate_type": gate_name,
                    "matrix_name": matrix_name,
                    "gradient_norm": float(np.linalg.norm(grad_matrix)),
                    "gradient_mean": float(np.mean(grad_matrix)),
                    "gradient_std": float(np.std(grad_matrix)),
                    "parent_event": "BackwardPass"
                })
    
    def log_weight_update_details(self, trace, iteration, model, Wf_prev):
        params = model.get_params()
        
        for matrix_name in self.weight_matrices:
            if matrix_name in params:
                current_matrix = params[matrix_name]
                
                if matrix_name == 'W_f':
                    weight_change = current_matrix - Wf_prev
                else:
                    weight_change_norm = float(np.linalg.norm(current_matrix))
                    weight_mean = float(np.mean(current_matrix))
                    weight_std = float(np.std(current_matrix))
                    
                    self.log.add_event(trace, f"WeightUpdate_{matrix_name}", attrs={
                        "iteration": iteration,
                        "matrix_name": matrix_name,
                        "weight_norm": weight_change_norm,
                        "weight_mean": weight_mean,
                        "weight_std": weight_std,
                        "weight_min": float(np.min(current_matrix)),
                        "weight_max": float(np.max(current_matrix)),
                        "parent_event": "WeightsUpdate"
                    })
                    continue
                
                change_norm = float(np.linalg.norm(weight_change))
                change_mean = float(np.mean(weight_change))
                change_std = float(np.std(weight_change))
                
                self.log.add_event(trace, f"WeightUpdate_{matrix_name}", attrs={
                    "iteration": iteration,
                    "matrix_name": matrix_name,
                    "weight_change_norm": change_norm,
                    "weight_change_mean": change_mean,
                    "weight_change_std": change_std,
                    "weight_change_min": float(np.min(weight_change)),
                    "weight_change_max": float(np.max(weight_change)),
                    "current_weight_norm": float(np.linalg.norm(current_matrix)),
                    "parent_event": "WeightsUpdate"
                })
        
        for bias_name in self.bias_vectors:
            if bias_name in params:
                current_bias = params[bias_name]
                bias_norm = float(np.linalg.norm(current_bias))
                bias_mean = float(np.mean(current_bias))
                bias_std = float(np.std(current_bias))
                
                self.log.add_event(trace, f"BiasUpdate_{bias_name}", attrs={
                    "iteration": iteration,
                    "bias_name": bias_name,
                    "bias_norm": bias_norm,
                    "bias_mean": bias_mean,
                    "bias_std": bias_std,
                    "bias_min": float(np.min(current_bias)),
                    "bias_max": float(np.max(current_bias)),
                    "parent_event": "WeightsUpdate"
                })