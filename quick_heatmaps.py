#!/usr/bin/env python3
"""
Quick script to generate L1 heatmaps for beta vs rho.
Extracts the necessary functions from create_paper_figures.py to display the heatmaps.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Setup figure parameters
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['lines.linewidth'] = 2.5
rcParams['lines.markersize'] = 8
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 11
rcParams['legend.fontsize'] = 11
rcParams['figure.titlesize'] = 14
rcParams['figure.dpi'] = 100

# Constants
RESULTS_BASE_DIR = './results'
L1_DIR_NAME = 'level1_final_results'
L2_DIR_NAME = 'level2_final_results'
OUTPUT_DIR = './quick_heatmaps'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_2d_sweep_data(level_dir):
    """Loads the 2D sweep NPZ file."""
    npz_path = os.path.join(level_dir, '2d_sweep_analysis', '2d_sweep_results.npz')
    print(f"\n--- Loading 2D data from '{npz_path}' ---")
    
    if os.path.exists(npz_path):
        try:
            data = np.load(npz_path)
            print("  Loaded 2D sweep data.")
            p1_name = str(data['param1_name'])
            p2_name = str(data['param2_name'])
            
            # Check for required keys
            required_keys = ['param1_values', 'param2_values', 'mean_fitness', 'mean_value']
            if not all(key in data for key in required_keys):
                print(f"  Error: Missing one or more required keys in {npz_path}: {required_keys}")
                return None
                
            return {
                'param1_name': p1_name, 'param1_values': data['param1_values'],
                'param2_name': p2_name, 'param2_values': data['param2_values'],
                'mean_fitness': data['mean_fitness'],
                'mean_value': data['mean_value'],
                'mean_deceptive': data.get('mean_deceptive', None)
            }
        except Exception as e:
            print(f"  Error loading {npz_path}: {e}")
            return None
    else:
        print(f"  Warning: NPZ file not found at {npz_path}")
        return None

def create_heatmap(x_values, y_values, z_values, 
                   xlabel=None, ylabel=None, title=None,
                   cmap='viridis', figsize=(8, 6.5),
                   colorbar_label=None, vmin=None, vmax=None, ax=None,
                   xticklabels=None, yticklabels=None, annotate=False, grid=False):
    """Create a heatmap, assumes z_values[i, j] corresponds to y_values[i] and x_values[j]."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Ensure z_values is 2D numpy array
    z_values = np.asarray(z_values)
    if z_values.ndim != 2:
        print(f"Warning: Z_values for heatmap '{title}' is not 2D (shape: {z_values.shape}). Skipping.")
        ax.text(0.5, 0.5, 'Invalid Data Shape', ha='center', va='center', transform=ax.transAxes, color='red', fontweight='bold', fontsize=14)
        ax.set_title(title, fontweight='bold')
        return fig, ax
    
    # Handle cases where all data is NaN
    all_nan = np.all(np.isnan(z_values))
    if all_nan:
        print(f"Warning: All Z-values are NaN for heatmap '{title}'. Displaying empty plot.")
        vmin_eff, vmax_eff = 0, 1
    elif vmin is None and vmax is None:
        vmin_eff = np.nanmin(z_values)
        vmax_eff = np.nanmax(z_values)
        if np.isclose(vmin_eff, vmax_eff):
            vmin_eff -= 0.5
            vmax_eff += 0.5
    else:
        vmin_eff, vmax_eff = vmin, vmax

    heatmap = ax.imshow(z_values, cmap=cmap, vmin=vmin_eff, vmax=vmax_eff, aspect='auto', origin='lower',
                      interpolation='nearest')
    ax.grid(grid)
    cbar = fig.colorbar(heatmap, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label, fontweight='bold', fontsize=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontweight='bold')
    if title:
        ax.set_title(title, fontweight='bold', fontsize=14)

    x_ticks = np.arange(z_values.shape[1])
    y_ticks = np.arange(z_values.shape[0])
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontweight='bold')
    elif x_values is not None and len(x_values) == z_values.shape[1]:
        ax.set_xticklabels([f"{v:.2g}" for v in x_values], rotation=45, ha="right", fontweight='bold')
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels, fontweight='bold')
    elif y_values is not None and len(y_values) == z_values.shape[0]:
        ax.set_yticklabels([f"{v:.2g}" for v in y_values], fontweight='bold')

    if annotate and not all_nan:
        for i in range(z_values.shape[0]):
            for j in range(z_values.shape[1]):
                val = z_values[i, j]
                if not np.isnan(val):
                    norm_val = heatmap.norm(val)
                    bgcolor = heatmap.cmap(norm_val)
                    luminance = 0.299*bgcolor[0] + 0.587*bgcolor[1] + 0.114*bgcolor[2]
                    textcolor = 'white' if luminance < 0.5 else 'black'
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=textcolor,
                            fontsize=11, fontweight='bold')

    return fig, ax

def create_l1_heatmaps():
    """Create and save the L1 heatmaps for rho vs beta."""
    # Load L1 2D sweep data
    l1_dir = os.path.join(RESULTS_BASE_DIR, L1_DIR_NAME)
    l1_2d_data = load_2d_sweep_data(l1_dir)
    
    if l1_2d_data is None:
        print("Error: Could not load L1 2D sweep data.")
        return
    
    # Get the parameter names
    p1_name = l1_2d_data['param1_name']
    p2_name = l1_2d_data['param2_name']
    
    # Ensure parameter names match expected values (rho and beta)
    if not (('rho' in p1_name and 'beta' in p2_name) or ('rho' in p2_name and 'beta' in p1_name)):
        print(f"Warning: Expected parameters 'rho' and 'beta', but got {p1_name} and {p2_name}.")
    
    # If parameters are flipped, rearrange for consistency
    if 'rho' in p2_name and 'beta' in p1_name:
        # Swap parameter names and values
        l1_2d_data['param1_name'], l1_2d_data['param2_name'] = l1_2d_data['param2_name'], l1_2d_data['param1_name']
        l1_2d_data['param1_values'], l1_2d_data['param2_values'] = l1_2d_data['param2_values'], l1_2d_data['param1_values']
        
        # Transpose data arrays for consistency
        l1_2d_data['mean_fitness'] = l1_2d_data['mean_fitness'].T
        l1_2d_data['mean_value'] = l1_2d_data['mean_value'].T
        if l1_2d_data['mean_deceptive'] is not None:
            l1_2d_data['mean_deceptive'] = l1_2d_data['mean_deceptive'].T
    
    # Now data should be in the format: param1=rho (rows/y-axis), param2=beta (cols/x-axis)
    rho_values = l1_2d_data['param1_values']
    beta_values = l1_2d_data['param2_values']
    mean_fitness = l1_2d_data['mean_fitness']
    mean_value = l1_2d_data['mean_value']
    
    # Compute Value/Fitness ratio with smart normalization
    # Use a small epsilon to avoid division by zero, and handle negative values
    epsilon = 1e-6
    # Create a shifted normalized ratio that preserves the sign of value
    # but doesn't have division issues when value is near zero
    ratio = np.zeros_like(mean_value)
    
    for i in range(mean_value.shape[0]):
        for j in range(mean_value.shape[1]):
            v = mean_value[i, j]
            f = mean_fitness[i, j]
            # Only compute ratio for non-NaN values
            if not np.isnan(v) and not np.isnan(f):
                # When value is negative, use a different approach to preserve sign
                if v < 0:
                    # For negative values, we want ratio to be negative but proportional to magnitude
                    ratio[i, j] = -1 * abs(v) / (f + epsilon)
                else:
                    # For positive values, standard ratio
                    ratio[i, j] = v / (f + epsilon)
            else:
                ratio[i, j] = np.nan
    
    # Create labels for axes
    rho_labels = [f"{v:.2g}" for v in rho_values]
    beta_labels = [f"{v:.2g}" for v in beta_values]
    
    # Transpose the data matrices to swap axes (now beta on y-axis, rho on x-axis)
    mean_fitness_t = mean_fitness.T
    mean_value_t = mean_value.T
    ratio_t = ratio.T
    
    # Create figure for fitness
    fig_fitness, ax_fitness = plt.subplots(figsize=(10, 8))
    create_heatmap(rho_values, beta_values, mean_fitness_t, ax=ax_fitness, annotate=True,
                   xlabel="Global Correlation (ρ)", ylabel="Selection Pressure (β)", 
                   title="L1: Final Fitness Heatmap (ρ vs β)",
                   cmap='viridis', colorbar_label="Mean Final Fitness",
                   xticklabels=rho_labels, yticklabels=beta_labels)
    fig_fitness.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "l1_beta_rho_fitness_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # Create figure for value
    fig_value, ax_value = plt.subplots(figsize=(10, 8))
    # Calculate robust vmin/vmax for RdBu_r centered around 0
    val_data_clean = mean_value_t[~np.isnan(mean_value_t)]
    val_max_abs = np.max(np.abs(val_data_clean)) if val_data_clean.size > 0 else 1.0
    vmin_val, vmax_val = -val_max_abs, val_max_abs
    create_heatmap(rho_values, beta_values, mean_value_t, ax=ax_value, annotate=True,
                   xlabel="Global Correlation (ρ)", ylabel="Selection Pressure (β)", 
                   title="L1: Final True Value Heatmap (ρ vs β)",
                   cmap='RdBu_r', vmin=vmin_val, vmax=vmax_val, colorbar_label="Mean Final True Value",
                   xticklabels=rho_labels, yticklabels=beta_labels)
    fig_value.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "l1_beta_rho_value_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # Create combined figure with both metrics
    fig_combined, axs = plt.subplots(1, 2, figsize=(16, 7))
    create_heatmap(rho_values, beta_values, mean_fitness_t, ax=axs[0], annotate=True,
                   xlabel="Global Correlation (ρ)", ylabel="Selection Pressure (β)", 
                   title="Final Fitness",
                   cmap='viridis', colorbar_label="Mean Final Fitness",
                   xticklabels=rho_labels, yticklabels=beta_labels)
    create_heatmap(rho_values, beta_values, mean_value_t, ax=axs[1], annotate=True,
                   xlabel="Global Correlation (ρ)", ylabel="Selection Pressure (β)", 
                   title="Final True Value",
                   cmap='RdBu_r', vmin=vmin_val, vmax=vmax_val, colorbar_label="Mean Final True Value",
                   xticklabels=rho_labels, yticklabels=beta_labels)
    fig_combined.suptitle("L1: Effects of ρ vs β on Fitness and True Value", fontsize=16, fontweight='bold')
    fig_combined.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(OUTPUT_DIR, "l1_beta_rho_combined_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # Create figure for Value/Fitness ratio
    fig_ratio, ax_ratio = plt.subplots(figsize=(10, 8))
    # Calculate robust vmin/vmax for ratio (diverging colormap centered around 0)
    ratio_data_clean = ratio_t[~np.isnan(ratio_t)]
    ratio_max_abs = np.max(np.abs(ratio_data_clean)) if ratio_data_clean.size > 0 else 1.0
    vmin_ratio, vmax_ratio = -ratio_max_abs, ratio_max_abs
    
    create_heatmap(rho_values, beta_values, ratio_t, ax=ax_ratio, annotate=True,
                   xlabel="Global Correlation (ρ)", ylabel="Selection Pressure (β)", 
                   title="L1: Value/Fitness Ratio Heatmap (ρ vs β)",
                   cmap='RdBu_r', vmin=vmin_ratio, vmax=vmax_ratio, 
                   colorbar_label="Value/Fitness Ratio",
                   xticklabels=rho_labels, yticklabels=beta_labels)
    
    fig_ratio.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "l1_beta_rho_value_fitness_ratio_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # Create combined figure with all three metrics (fitness, value, and ratio)
    fig_all, axs = plt.subplots(1, 3, figsize=(22, 7))
    
    create_heatmap(rho_values, beta_values, mean_fitness_t, ax=axs[0], annotate=True,
                   xlabel="Global Correlation (ρ)", ylabel="Selection Pressure (β)", 
                   title="Final Fitness",
                   cmap='viridis', colorbar_label="Mean Final Fitness",
                   xticklabels=rho_labels, yticklabels=beta_labels)
    
    create_heatmap(rho_values, beta_values, mean_value_t, ax=axs[1], annotate=True,
                   xlabel="Global Correlation (ρ)", ylabel="Selection Pressure (β)", 
                   title="Final True Value",
                   cmap='RdBu_r', vmin=vmin_val, vmax=vmax_val, colorbar_label="Mean Final True Value",
                   xticklabels=rho_labels, yticklabels=beta_labels)
    
    create_heatmap(rho_values, beta_values, ratio_t, ax=axs[2], annotate=True,
                   xlabel="Global Correlation (ρ)", ylabel="Selection Pressure (β)", 
                   title="Value/Fitness Ratio",
                   cmap='RdBu_r', vmin=vmin_ratio, vmax=vmax_ratio, colorbar_label="Value/Fitness Ratio",
                   xticklabels=rho_labels, yticklabels=beta_labels)
    
    fig_all.suptitle("L1: Effects of ρ vs β on Fitness, Value, and Value/Fitness Ratio", fontsize=16, fontweight='bold')
    fig_all.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(OUTPUT_DIR, "l1_beta_rho_all_metrics_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def create_l2_heatmaps():
    """Create and save the L2 heatmaps for beta vs deceptive_prop."""
    # Load L2 2D sweep data
    l2_dir = os.path.join(RESULTS_BASE_DIR, L2_DIR_NAME)
    l2_2d_data = load_2d_sweep_data(l2_dir)
    
    if l2_2d_data is None:
        print("Error: Could not load L2 2D sweep data.")
        return
    
    # Get the parameter names
    p1_name = l2_2d_data['param1_name']
    p2_name = l2_2d_data['param2_name']
    
    # Ensure parameter names match expected values (beta and deceptive_prop)
    if not (('beta' in p1_name and 'deceptive_prop' in p2_name) or ('beta' in p2_name and 'deceptive_prop' in p1_name)):
        print(f"Warning: Expected parameters 'beta' and 'deceptive_prop', but got {p1_name} and {p2_name}.")
    
    # If parameters are flipped, rearrange for consistency to match create_paper_figures.py behavior
    if p1_name == "deceptive_prop" and p2_name == "beta":
        # Swap parameter names and values
        l2_2d_data['param1_name'], l2_2d_data['param2_name'] = l2_2d_data['param2_name'], l2_2d_data['param1_name']
        l2_2d_data['param1_values'], l2_2d_data['param2_values'] = l2_2d_data['param2_values'], l2_2d_data['param1_values']
        
        # Transpose data arrays for consistency
        l2_2d_data['mean_fitness'] = l2_2d_data['mean_fitness'].T
        l2_2d_data['mean_value'] = l2_2d_data['mean_value'].T
        if l2_2d_data['mean_deceptive'] is not None:
            l2_2d_data['mean_deceptive'] = l2_2d_data['mean_deceptive'].T
    
    # Now data should be in the format: param1=beta (rows/y-axis), param2=deceptive_prop (cols/x-axis)
    beta_values = l2_2d_data['param1_values']
    deceptive_prop_values = l2_2d_data['param2_values']
    mean_fitness = l2_2d_data['mean_fitness']
    mean_value = l2_2d_data['mean_value']
    mean_deceptive = l2_2d_data['mean_deceptive']
    
    # Compute Value/Fitness ratio with smart normalization
    # Use a small epsilon to avoid division by zero, and handle negative values
    epsilon = 1e-6
    # Create a shifted normalized ratio that preserves the sign of value
    # but doesn't have division issues when value is near zero
    ratio = np.zeros_like(mean_value)
    
    for i in range(mean_value.shape[0]):
        for j in range(mean_value.shape[1]):
            v = mean_value[i, j]
            f = mean_fitness[i, j]
            # Only compute ratio for non-NaN values
            if not np.isnan(v) and not np.isnan(f):
                # When value is negative, use a different approach to preserve sign
                if v < 0:
                    # For negative values, we want ratio to be negative but proportional to magnitude
                    ratio[i, j] = -1 * abs(v) / (f + epsilon)
                else:
                    # For positive values, standard ratio
                    ratio[i, j] = v / (f + epsilon)
            else:
                ratio[i, j] = np.nan
    
    # Create labels for axes
    beta_labels = [f"{v:.2g}" for v in beta_values]
    deceptive_prop_labels = [f"{v:.2g}" for v in deceptive_prop_values]
    
    # Create figure for fitness
    fig_fitness, ax_fitness = plt.subplots(figsize=(10, 8))
    create_heatmap(deceptive_prop_values, beta_values, mean_fitness, ax=ax_fitness, annotate=True,
                   xlabel="Initial Deceptive Proportion", ylabel="Selection Pressure (β)", 
                   title="L2: Final Fitness Heatmap",
                   cmap='viridis', colorbar_label="Mean Final Fitness",
                   xticklabels=deceptive_prop_labels, yticklabels=beta_labels)
    fig_fitness.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "l2_beta_deceptive_fitness_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # Create figure for value
    fig_value, ax_value = plt.subplots(figsize=(10, 8))
    # Calculate robust vmin/vmax for RdBu_r centered around 0
    val_data_clean = mean_value[~np.isnan(mean_value)]
    val_max_abs = np.max(np.abs(val_data_clean)) if val_data_clean.size > 0 else 1.0
    vmin_val, vmax_val = -val_max_abs, val_max_abs
    create_heatmap(deceptive_prop_values, beta_values, mean_value, ax=ax_value, annotate=True,
                   xlabel="Initial Deceptive Proportion", ylabel="Selection Pressure (β)", 
                   title="L2: Final True Value Heatmap",
                   cmap='RdBu_r', vmin=vmin_val, vmax=vmax_val, colorbar_label="Mean Final True Value",
                   xticklabels=deceptive_prop_labels, yticklabels=beta_labels)
    fig_value.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "l2_beta_deceptive_value_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # Create figure for deceptive ratio if available
    if mean_deceptive is not None:
        fig_deceptive, ax_deceptive = plt.subplots(figsize=(10, 8))
        # Calculate robust vmax for deceptive ratio (vmin is 0)
        decep_data_clean = mean_deceptive[~np.isnan(mean_deceptive)]
        vmax_dec = np.max(decep_data_clean) if decep_data_clean.size > 0 else 1.0
        create_heatmap(deceptive_prop_values, beta_values, mean_deceptive, ax=ax_deceptive, annotate=True,
                      xlabel="Initial Deceptive Proportion", ylabel="Selection Pressure (β)", 
                      title="L2: Final Deceptive Ratio Heatmap",
                      cmap='Reds', vmin=0, vmax=max(0.01, vmax_dec), colorbar_label="Mean Final Deceptive Ratio",
                      xticklabels=deceptive_prop_labels, yticklabels=beta_labels)
        fig_deceptive.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, "l2_beta_deceptive_ratio_heatmap.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    # Create figure for Value/Fitness ratio
    fig_ratio, ax_ratio = plt.subplots(figsize=(10, 8))
    # Calculate robust vmin/vmax for ratio (diverging colormap centered around 0)
    ratio_data_clean = ratio[~np.isnan(ratio)]
    ratio_max_abs = np.max(np.abs(ratio_data_clean)) if ratio_data_clean.size > 0 else 1.0
    vmin_ratio, vmax_ratio = -ratio_max_abs, ratio_max_abs
    
    create_heatmap(deceptive_prop_values, beta_values, ratio, ax=ax_ratio, annotate=True,
                   xlabel="Initial Deceptive Proportion", ylabel="Selection Pressure (β)", 
                   title="L2: Value/Fitness Ratio Heatmap",
                   cmap='RdBu_r', vmin=vmin_ratio, vmax=vmax_ratio, 
                   colorbar_label="Value/Fitness Ratio",
                   xticklabels=deceptive_prop_labels, yticklabels=beta_labels)
    
    fig_ratio.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "l2_beta_deceptive_value_fitness_ratio_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # Create combined figure with all metrics (fitness, value, deceptive ratio, and value/fitness ratio)
    n_metrics = 4 if mean_deceptive is not None else 3
    fig_width = n_metrics * 7
    fig_all, axs = plt.subplots(1, n_metrics, figsize=(fig_width, 7))
    
    # Plot Fitness
    create_heatmap(deceptive_prop_values, beta_values, mean_fitness, ax=axs[0], annotate=True,
                   xlabel="Initial Deceptive Proportion", ylabel="Selection Pressure (β)", 
                   title="Final Fitness",
                   cmap='viridis', colorbar_label="Mean Final Fitness",
                   xticklabels=deceptive_prop_labels, yticklabels=beta_labels)
    
    # Plot Value
    create_heatmap(deceptive_prop_values, beta_values, mean_value, ax=axs[1], annotate=True,
                   xlabel="Initial Deceptive Proportion", ylabel="Selection Pressure (β)", 
                   title="Final True Value",
                   cmap='RdBu_r', vmin=vmin_val, vmax=vmax_val, colorbar_label="Mean Final True Value",
                   xticklabels=deceptive_prop_labels, yticklabels=beta_labels)
    
    # If deceptive ratio data is available, add it to the combined plot
    idx = 2
    if mean_deceptive is not None:
        create_heatmap(deceptive_prop_values, beta_values, mean_deceptive, ax=axs[idx], annotate=True,
                      xlabel="Initial Deceptive Proportion", ylabel="Selection Pressure (β)", 
                      title="Final Deceptive Ratio",
                      cmap='Reds', vmin=0, vmax=max(0.01, vmax_dec), colorbar_label="Mean Final Deceptive Ratio",
                      xticklabels=deceptive_prop_labels, yticklabels=beta_labels)
        idx += 1
    
    # Plot Value/Fitness Ratio
    create_heatmap(deceptive_prop_values, beta_values, ratio, ax=axs[idx], annotate=True,
                   xlabel="Initial Deceptive Proportion", ylabel="Selection Pressure (β)", 
                   title="Value/Fitness Ratio",
                   cmap='RdBu_r', vmin=vmin_ratio, vmax=vmax_ratio, colorbar_label="Value/Fitness Ratio",
                   xticklabels=deceptive_prop_labels, yticklabels=beta_labels)
    
    fig_all.suptitle("L2: Effects of Initial Deceptive Proportion vs β on Model Outcomes", fontsize=16, fontweight='bold')
    fig_all.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(OUTPUT_DIR, "l2_beta_deceptive_all_metrics_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    print("Generating quick heatmaps for L1 and L2 data...")
    create_l1_heatmaps()
    create_l2_heatmaps()
    print("Done!") 