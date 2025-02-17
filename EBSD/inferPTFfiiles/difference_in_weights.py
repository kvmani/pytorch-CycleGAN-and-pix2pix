import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def compare_weights(checkpoint1_path, checkpoint2_path):
    """
    Loads two checkpoints of the same network (e.g., netG_A),
    calculates the average absolute difference of each layer's weights,
    returns:
        differences: a sorted list of (layer_name, diff) in descending order of diff
        max_layer: the layer_name with the maximum difference
        min_layer: the layer_name with the minimum difference
    """
    # Load state dicts
    state1 = torch.load(checkpoint1_path, map_location='cpu')
    state2 = torch.load(checkpoint2_path, map_location='cpu')

    differences = []

    weight_keys = [k for k in state1.keys() if k.endswith(".weight")]

    for layer_name in weight_keys:
        if layer_name in state2:
            w1 = state1[layer_name].numpy()
            w2 = state2[layer_name].numpy()
            
            # Compute mean absolute difference
            layer_diff = np.mean(np.abs(w2 - w1))
            differences.append((layer_name, layer_diff))
    
    # Sort by descending difference
    differences.sort(key=lambda x: x[1], reverse=True)
    
    # The layer with the largest difference is the first in the sorted list
    max_layer = differences[0][0]
    # The layer with the smallest difference is the last in the sorted list
    min_layer = differences[-1][0]
    
    return differences, max_layer, min_layer

def plot_layer_distributions(layer_name, checkpoints, layer_label, bins=50):
    """
    Plots the weight distribution of 'layer_name' for each checkpoint in 'checkpoints'.
    'layer_label' is used for the plot title (e.g. "Max Layer" or "Min Layer").
    'bins' is the histogram bins.
    """
    plt.figure(figsize=(8,6))
    
    for ckpt_path in checkpoints:
        # Extract epoch info from filename (optional, if you want to label)
        # e.g. "100_net_G_A.pth" -> epoch 100
        base = os.path.basename(ckpt_path)
        # Attempt to parse epoch from the filename (assuming something like "100_net_G_A.pth")
        epoch_label = base.split("_")[0]  # "100"
        
        # Load checkpoint
        state_dict = torch.load(ckpt_path, map_location='cpu')
        weights = state_dict[layer_name].numpy().flatten()
        
        # Plot histogram or KDE
        # For a more direct comparison, use hist density:
        plt.hist(weights, bins=bins, alpha=0.4, density=True, label=f"epoch {epoch_label}")

    plt.title(f"Weight Distribution ({layer_label} = {layer_name})")
    plt.xlabel("Weight Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1) Two reference checkpoints to find the max/min difference layers
    checkpoint1 = "/home/lus04/kvmani/ml_works/kaushal_2025/cyclegan/checkpoints/ebsd_data_2.0_lr1e-4_pool_100_batch_32_vanilla/50_net_G_A.pth"
    checkpoint2 = "/home/lus04/kvmani/ml_works/kaushal_2025/cyclegan/checkpoints/ebsd_data_2.0_lr1e-4_pool_100_batch_32_vanilla/600_net_G_A.pth"
    
    differences, max_layer, min_layer = compare_weights(checkpoint1, checkpoint2)
    
    # Print top 5 biggest changes
    print("Top layers by weight change:")
    for name, diff in differences[:5]:
        print(f"{name}: {diff:.4f}")
    print(f"\nMax Layer: {max_layer}")
    print(f"Min Layer: {min_layer}")
    
    # 2) Now choose the intervals at which to compare weight distributions.
    # Suppose total epochs = 600 and step = 100
    # We'll gather the checkpoint paths at 0, 100, 200, 300, 400, 500, 600.
    # (Adjust this logic to match your real checkpoint naming.)
    
    # Example naming pattern: "0_net_G_A.pth", "100_net_G_A.pth", ...
    interval = 100
    max_epoch = 600
    # If there's a "0_net_G_A.pth", optionally include it. 
    # Or if your training starts from 50, adjust accordingly.
    epochs = list(range(100, max_epoch+1, interval))  # [0,100,200,300,400,500,600]
    
    # Build a list of checkpoint paths
    # NOTE: Modify path pattern as needed to match your actual filenames
    base_dir = "/home/lus04/kvmani/ml_works/kaushal_2025/cyclegan/checkpoints/ebsd_data_2.0_lr1e-4_pool_100_batch_32_vanilla"
    ckpts_for_plot = []
    for e in epochs:
        ckpt_file = f"{e}_net_G_A.pth"
        ckpt_path = os.path.join(base_dir, ckpt_file)
        if os.path.exists(ckpt_path):
            ckpts_for_plot.append(ckpt_path)
        else:
            print(f"Warning: {ckpt_path} does not exist, skipping.")
    
    # 3) Plot distribution for the max_layer at the selected intervals
    plot_layer_distributions(max_layer, ckpts_for_plot, layer_label="Max Layer", bins=60)
    
    # 4) Plot distribution for the min_layer at the selected intervals
    plot_layer_distributions(min_layer, ckpts_for_plot, layer_label="Min Layer", bins=60)
