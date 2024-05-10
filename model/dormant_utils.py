import random
import re
import time

from copy import deepcopy
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

class LinearOutputHook:
    def __init__(self):
        self.outputs = []
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

def cal_dormant_ratio(model, *inputs, type='policy', percentage=0.1):
    metrics = dict()
    hooks = []
    hook_handlers = []
    total_neurons = 0
    dormant_neurons = 0
    dormant_indices = dict()
    active_indices = dict()

    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = LinearOutputHook()
            hooks.append(hook)
            hook_handlers.append(module.register_forward_hook(hook))

    with torch.no_grad():
        model(*inputs)

    count = 0
    for module, hook in zip(
        (module
         for module in model.modules() if isinstance(module, nn.Linear)),
            hooks):
        with torch.no_grad():
            for output_data in hook.outputs:
                mean_output = output_data.abs().mean(0)
                avg_neuron_output = mean_output.mean()
                dormant_indice = (mean_output < avg_neuron_output *
                                   percentage).nonzero(as_tuple=True)[0]
                all_indice = list(range(module.weight.shape[0]))
                active_indice = [index for index in all_indice if index not in dormant_indice]
                total_neurons += module.weight.shape[0]
                dormant_neurons += len(dormant_indice)
                module_dormant_ratio = len(dormant_indices) / module.weight.shape[0]
                #metrics[
                    #type + '_'+ str(count) +
                    #'_dormant'] = module_dormant_ratio
                if module_dormant_ratio > 0.1:
                    dormant_indices[str(count)] = dormant_indice
                    active_indices[str(count)] = active_indice
                count += 1

    for hook in hooks:
        hook.outputs.clear()

    for hook_handler in hook_handlers:
        hook_handler.remove()

    metrics[type + "_output_dormant_ratio"] = dormant_neurons / total_neurons

    return metrics, dormant_indices, active_indices

def cal_dormant_grad(model, type = 'policy', percentage=0.025):
    metrics = dict()
    total_neurons = 0
    dormant_neurons = 0
    dormant_indices = dict()
    active_indices = dict()
    
    count = 0
    for module in (module for module in model.modules() if isinstance(module, nn.Linear)):
        grad_norm = module.weight.grad.norm(dim=1)  
        avg_grad_norm = grad_norm.mean()
        dormant_indice = (grad_norm < avg_grad_norm * percentage).nonzero(as_tuple=True)[0]
        total_neurons += module.weight.shape[0]
        dormant_neurons += len(dormant_indice)
        module_dormant_grad = len(dormant_indice) / module.weight.shape[0]
        metrics[
                type + '_' + str(count) +
                '_grad_dormant'] = module_dormant_grad
        count += 1
    metrics[type + "_grad_dormant_ratio"] = dormant_neurons / total_neurons
    return metrics

def perturb(net, optimizer, perturb_factor):
    linear_keys = [
        name for name, mod in net.named_modules()
        if isinstance(mod, torch.nn.Linear)
    ]
    new_net = deepcopy(net)
    new_net.apply(weight_init)

    for name, param in net.named_parameters():
        if any(key in name for key in linear_keys):
            noise = new_net.state_dict()[name] * (1 - perturb_factor)
            param.data = param.data * perturb_factor + noise
        else:
            param.data = net.state_dict()[name]
    optimizer.state = defaultdict(dict)
    return net, optimizer

def dormant_perturb(model, optimizer, dormant_indices, perturb_factor=0.2):
    random_model = deepcopy(model)
    random_model.apply(weight_init)
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    random_layers = [module for module in random_model.modules() if isinstance(module, nn.Linear)]

    for key in dormant_indices:
        perturb_layer = linear_layers[key]
        random_layer = random_layers[key]
        with torch.no_grad():
            for index in dormant_indices[key]:
                noise = (random_layer.weight[index, :] * (1 - perturb_factor)).clone()
                perturb_layer.weight[index, :] = perturb_layer.weight[index, :] * perturb_factor + noise

    optimizer.state = defaultdict(dict)
    return model, optimizer
    
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def neurons_impact(model, *input, indices, perturb_factor):
    """
    Calculate the impact of neurons in the model.

    Args:
        model (nn.Module): The neural network model.
        indices (dict): A dictionary where keys are layer indices and values are lists of neuron indices.
        perturb_factor (float): The perturbation factor.

    Returns:
        dict: A dictionary where keys are layer indices and values are impact arrays.
    """
    if not indices:
        return {}
    impact = dict()

    perturb_model = deepcopy(model)
    random_model = deepcopy(model)
    random_model.apply(weight_init)

    perturb_layers = [module for module in perturb_model.modules() if isinstance(module, nn.Linear)]
    random_layers = [module for module in random_model.modules() if isinstance(module, nn.Linear)]
    
    for key in indices:
        perturb_layer = perturb_layers[key]
        random_layer = random_layers[key]
        with torch.no_grad():
            for index in indices[key]:
                noise = (random_layer.weight[index, :] * (1 - perturb_factor)).clone()
                perturb_layer.weight[index, :] = perturb_layer.weight[index, :] * perturb_factor + noise
            perturb_output, _ = perturb_model(*input)
            ori_output, _ = model(*input)

        impact[key] = (perturb_output - ori_output).mean(0).cpu().numpy()
        perturb_model = deepcopy(model)
        perturb_layers = [module for module in perturb_model.modules() if isinstance(module, nn.Linear)]

    return impact

def normalize(data):
    min_val = data.min()
    max_val = data.max()
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def correlation(causal_weight, dormant_impact, active_impact):
    metrics = dict()
    if dormant_impact:
        for key in dormant_impact:
            dormant_correlation = np.corrcoef(dormant_impact[key], causal_weight)[0, 1]
            metrics['dormant_correlation_' + str(key)] = dormant_correlation
    if active_impact:
        for key in active_impact:
            active_correlation = np.corrcoef(active_impact[key], causal_weight)[0, 1]
            metrics['active_correlation_' + str(key)] = active_correlation
    if not dormant_impact and not active_impact:
        return metrics
    return metrics

def process(data):
    mean = data.mean()
    processed_data = data / mean
    return processed_data

def similarity(causal_weight, dormant_impact, active_impact):
    metrics = dict()
    if not dormant_impact or active_impact:
        return {}
    causal_weight = process(causal_weight).reshape(1, -1)
    for key in dormant_impact:
        dormant_vector = process(np.abs(dormant_impact[key])).reshape(1, -1)
        active_vector = process(np.abs(active_impact[key])).reshape(1, -1)
        causal_norm = np.linalg.norm(causal_weight)
        dormant_norm = np.linalg.nrom(dormant_vector)
        active_norm = np.linalg.nrom(active_vector)
        if causal_norm!=0 and dormant_norm!=0:
            metrics['dormant_' + str(key) + '_similarity'] = cosine_similarity(causal_weight, dormant_vector)
        
        if causal_norm!=0 and active_norm!=0:
            metrics['active_' + str(key) + '_similarity'] = cosine_similarity(causal_weight, active_vector)

    return metrics

def perturb_factor(dormant_ratio, max_perturb_factor=0.9, min_perturb_factor=0.2):
    return min(max(min_perturb_factor, 1 - dormant_ratio), max_perturb_factor)
