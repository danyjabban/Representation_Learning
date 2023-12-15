import torch
import torch.nn as nn
import numpy as np

"""
Pruning code from ECE 661 homework 4, written by Michael Li
"""
# def prune_by_percentage(layer, q, device):
#     """
#     Pruning the weight paramters by threshold.
#     :param q: pruning percentile. 'q' percent of the least 
#     significant weight parameters will be pruned.
#     """
#     with torch.no_grad():
#         # Convert the weight of "layer" to numpy array
#         layer_weight = layer.weight.detach().cpu().numpy()
#         # Compute the q-th percentile of the abs of the converted array
#         percentile = np.percentile(np.abs(layer_weight.flatten()), q)
#         # Generate a binary mask same shape as weight to decide which element to prune
#         masked_obj = np.ma.masked_greater_equal(x=np.abs(layer_weight), value=percentile, copy=True)
#         mask_int = np.ma.getmask(masked_obj).astype(int)
#         # Convert mask to torch tensor and put on GPU
#         mask_tensor = torch.tensor(mask_int).to(device)
#         # Multiply the weight by mask to perform pruning
#         assert mask_int.shape == layer_weight.shape
#         # layer.weight.data = mask_tensor * layer.weight.data
#         layer.weight.data = layer.weight.data.clone().detach().requires_grad_(True) * mask_tensor
#     return


# def prune_net(net, q_val, device, verbose=False):
#     print("pruning net")
#     for name, layer in net.named_modules():
#         if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)) and 'downsample' not in name:
#             # change q value
#             prune_by_percentage(layer, q=q_val, device=device)
#             # break
#             ## Optional: Check the sparsity you achieve in each layer
#             ## Convert the weight of "layer" to numpy array
#             np_weight = layer.weight.detach().cpu().numpy()
#             ## Count number of zeros
#             zeros = sum((np_weight == 0).flatten())
#             ## Count number of parameters
#             total = len(np_weight.flatten())
#             ## Print sparsity
#             # print('Sparsity of ' + name + ': '+ str(zeros/total))
#             if verbose:
#                 print('Sparsity of %s: %g' % (name, zeros/total))
#     return


def check_pruned_net(net, verbose=False):
    print("checking how pruned the net is")
    sparsity_list = []
    x_ticks_labels = []
    zeros_sum = 0
    total_sum = 0
    for name, layer in net.named_modules():
        if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)) and 'downsample' not in name:
            # Your code here:
            # Convert the weight of "layer" to numpy array
            np_weight = layer.weight.detach().cpu().numpy()
            # Count number of zeros
            zeros = sum((np_weight == 0).flatten())
            # Count number of parameters
            total = len(np_weight.flatten())
            zeros_sum += zeros
            total_sum += total
            sparsity_list.append(zeros / (total / 1.0))
            x_ticks_labels.append(name)
            if verbose: print('Sparsity of %s: %g' % (name, zeros / (total / 1.0)))   
    print('Total sparsity: %g' % (zeros_sum / (total_sum / 1.0)))
    return


def global_prune_by_percentage(net, q, device):
    """
    Pruning the weight paramters by threshold.
    :param q: pruning percentile. 'q' percent of the least 
    significant weight parameters will be pruned.
    """
    # A list to gather all the weights
    flattened_weights = []
    # Find global pruning threshold
    for name,layer in net.named_modules():
        if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)) and 'downsample' not in name:
            # Convert weight to numpy
            # Flatten the weight and append to flattened_weights
            flattened_weights.append(layer.weight.detach().cpu().numpy().flatten())
    
    # Concate all weights into a np array
    flattened_weights = np.concatenate(flattened_weights)
    # Find global pruning threshold
    threshold = np.percentile(np.abs(flattened_weights.flatten()), q)
    
    # Apply pruning threshold to all layers
    for name,layer in net.named_modules():
        if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)) and 'downsample' not in name:
            # Convert weight to numpy
            layer_weight = layer.weight.detach().cpu().numpy()
            
            # Generate a binary mask same shape as weight to decide which element to prune
            masked_obj = np.ma.masked_greater_equal(x=np.abs(layer_weight), value=threshold, copy=True)
            mask_int = np.ma.getmask(masked_obj).astype(int)
            # Convert mask to torch tensor and put on GPU
            mask_tensor = torch.tensor(mask_int).to(device)
            # Multiply the weight by mask to perform pruning
            layer.weight.data = layer.weight.data.clone().detach().requires_grad_(True) * mask_tensor
    return
