# -*- coding: utf-8 -*-
"""
Simulation of AI Model Evolution under Alignment Pressure.

Includes Levels 0-3 simulation, analysis, and visualization.
Workflow: Simulate -> Analyze (in memory) -> Visualize (from memory analysis).
Does not include logic for loading/visualizing previously saved *individual* run files
unless using --analyze_only flag.

NOW INCLUDES 2D SWEEP FUNCTIONALITY.
"""

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import time
import argparse
import os
import json
import datetime
import yaml
# from matplotlib.colors import ListedColormap # Not strictly needed unless customizing cmaps
import glob
import sys
import copy # Added for deep copying parameters


# ===== Visualization Utilities (Plotting Primitives) =====
# These functions create basic plot types based on input data

def save_figure(fig, filename, results_dir, dpi=300, bbox_inches='tight'):
    """Save a matplotlib figure with consistent settings."""
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved figure to {filepath}")
    except Exception as e:
        print(f"Error saving figure {filepath}: {e}")
    finally:
        plt.close(fig) # Close the figure to free memory

def create_errorbar_plot(x_values, y_values, yerr=None, xlabel=None, ylabel=None,
                         title=None, grid=True, color=None, marker='o',
                         capsize=5, figsize=(8, 6), annotations=None, ax=None):
    """Create an error bar plot, optionally on existing axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.errorbar(x_values, y_values, yerr=yerr, marker=marker,
                capsize=capsize, color=color, linestyle='-') # Ensure line connects markers

    if annotations:
        for idx, text in annotations.items():
            if idx < len(x_values):
                ax.annotate(text, (x_values[idx], y_values[idx]),
                            textcoords="offset points", xytext=(5, 5), ha='left')

    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    if grid: ax.grid(True, alpha=0.3)

    # Don't call tight_layout here if using external axes
    # if ax is None: fig.tight_layout()
    return fig, ax

def create_history_plot(history_values, xlabel='Generation', ylabel=None,
                        title=None, grid=True, color=None,
                        figsize=(8, 6), alpha=1.0, label=None, ax=None):
    """Create a line plot of a metric's history, optionally on existing axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(history_values, color=color, alpha=alpha, label=label)

    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    if grid: ax.grid(True, alpha=0.3)
    if label: ax.legend()

    return fig, ax

def create_multi_panel_plot(n_rows, n_cols, figsize=(16, 12), **kwargs):
    """Create a multi-panel plot figure and axes array."""
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, **kwargs)
    return fig, axs

def create_scatter_plot(x_values, y_values, size=None, color=None,
                        xlabel=None, ylabel=None, title=None,
                        grid=True, figsize=(8, 6), alpha=0.7,
                        cmap='viridis', annotations=None, colorbar_label=None, ax=None):
    """Create a scatter plot, optionally on existing axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    scatter = ax.scatter(x_values, y_values, s=size, c=color,
                         cmap=cmap, alpha=alpha)

    if color is not None and isinstance(color, (list, np.ndarray)) and cmap is not None:
        # Check if the color data requires a colorbar
        if np.isscalar(color) or all(c == color[0] for c in color):
            pass # Single color, no colorbar needed
        else:
             try:
                 cbar = fig.colorbar(scatter, ax=ax)
                 if colorbar_label:
                     cbar.set_label(colorbar_label)
             except Exception as e:
                 print(f"Warning: Could not create colorbar: {e}")


    if annotations:
        for idx, text in annotations.items():
            if idx < len(x_values):
                ax.annotate(text, (x_values[idx], y_values[idx]),
                            textcoords="offset points", xytext=(5, 5), ha='left')

    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    if grid: ax.grid(True, alpha=0.3)

    return fig, ax

def create_bar_plot(x_values, y_values, yerr=None,
                    xlabel=None, ylabel=None, title=None,
                    grid=True, figsize=(8, 6),
                    colors=None, alpha=0.7, xtick_labels=None,
                    horizontal=False, width=0.8, label=None, ax=None, bottom=None):
    """Create a bar plot, optionally on existing axes, supporting stacking."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if horizontal:
        bars = ax.barh(x_values, y_values, height=width, xerr=yerr, color=colors, alpha=alpha, label=label, left=bottom)
    else:
        bars = ax.bar(x_values, y_values, width=width, yerr=yerr, color=colors, alpha=alpha, label=label, bottom=bottom)

    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    if grid:
        ax.grid(True, axis='x' if horizontal else 'y', alpha=0.3)

    if xtick_labels:
        if horizontal:
            ax.set_yticks(x_values)
            ax.set_yticklabels(xtick_labels)
        else:
            ax.set_xticks(x_values)
            ax.set_xticklabels(xtick_labels)

    if label: ax.legend()

    return fig, ax

def create_heatmap(x_values, y_values, z_values,
                   xlabel=None, ylabel=None, title=None,
                   cmap='viridis', figsize=(10, 8),
                   colorbar_label=None, vmin=None, vmax=None, ax=None,
                   xticklabels=None, yticklabels=None, annotate=False, grid=False):
    """Create a heatmap, optionally on existing axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Use pcolormesh for potentially non-uniform grids or if x/y are edges
    # Requires x/y to define grid boundaries (len = N+1) or centers (len = N)
    # If using imshow, extent might be needed if x/y are not just indices
    heatmap = ax.imshow(z_values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', origin='lower')

    # Set grid visibility
    ax.grid(grid)

    cbar = fig.colorbar(heatmap, ax=ax)
    if colorbar_label: cbar.set_label(colorbar_label)

    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title)

    # Set ticks based on the shape of z_values
    x_ticks = np.arange(z_values.shape[1])
    y_ticks = np.arange(z_values.shape[0])
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Apply tick labels - use provided labels or format the coordinate values
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    elif x_values is not None and len(x_values) == z_values.shape[1]:
         ax.set_xticklabels([f"{v:.2g}" for v in x_values]) # Format coordinate values

    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    elif y_values is not None and len(y_values) == z_values.shape[0]:
        ax.set_yticklabels([f"{v:.2g}" for v in y_values]) # Format coordinate values

    # Annotation logic from sprawling code
    if annotate:
        for i in range(z_values.shape[0]):
            for j in range(z_values.shape[1]):
                val = z_values[i, j]
                if not np.isnan(val): # Only annotate non-NaN values
                     # Choose text color based on background brightness
                     bgcolor = heatmap.cmap(heatmap.norm(val))
                     brightness = sum(bgcolor[:3]) # Simple brightness check
                     textcolor = 'white' if brightness < 1.5 else 'black' # Threshold might need adjustment
                     ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=textcolor, fontsize=8)

    return fig, ax

def create_belief_space_plot(belief_values, belief_alignments, cluster_indices=None,
                             cluster_params=None, colors=None, alpha=0.6,
                             xlabel='True Value (v)', ylabel='Alignment (a)',
                             title='Belief Space Distribution',
                             figsize=(10, 8), highlight_indices=None,
                             highlight_marker='X', highlight_size=100,
                             highlight_alpha=0.8, shade_deceptive=True,
                             xlim=None, ylim=None, ax=None):
    """Create a plot of the belief space, optionally on existing axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if colors is None:
        # Use tab10 colormap by default
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in range(10)] # Get list of colors


    # Auto-determine limits if not provided
    val_min, val_max = np.nanmin(belief_values), np.nanmax(belief_values)
    align_min, align_max = np.nanmin(belief_alignments), np.nanmax(belief_alignments)
    if xlim is None:
        pad = max(1.0, (val_max - val_min) * 0.1) if val_max > val_min else 1.0
        xlim = (val_min - pad, val_max + pad)
    if ylim is None:
        pad = max(1.0, (align_max - align_min) * 0.1) if align_max > align_min else 1.0
        ylim = (align_min - pad, align_max + pad)

    # Plot by cluster or all points
    if cluster_indices is not None and cluster_params is not None:
        n_clusters = np.nanmax(cluster_indices) + 1
        for i in range(n_clusters):
            cluster_mask = (cluster_indices == i)
            cluster_name = cluster_params[i].get('name', f'Cluster {i}') if i < len(cluster_params) else f'Cluster {i}'
            ax.scatter(belief_values[cluster_mask], belief_alignments[cluster_mask],
                       alpha=alpha, color=colors[i % len(colors)], label=cluster_name, s=10)
        # Plot cluster means
        for i, cluster in enumerate(cluster_params):
             ax.scatter([cluster.get('mu_v', 0)], [cluster.get('mu_a', 0)], color=colors[i % len(colors)],
                       marker='x', s=150, linewidth=3, label=f'{cluster.get("name", f"Cluster {i}")} Mean', zorder=10)
    else:
        ax.scatter(belief_values, belief_alignments, alpha=alpha, color=colors[0], label='Beliefs', s=10)

    # Highlight points
    if highlight_indices is not None and len(highlight_indices) > 0:
        highlight_indices = np.asarray(highlight_indices)
        highlight_mask = np.isin(np.arange(len(belief_values)), highlight_indices)
        if np.any(highlight_mask):
             ax.scatter(belief_values[highlight_mask], belief_alignments[highlight_mask], alpha=highlight_alpha,
                       color='orange', marker=highlight_marker, s=highlight_size, edgecolors='black',
                       linewidth=0.5, label='Highlighted', zorder=5) # Simplified highlight color

    # Lines and shading
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.4, linewidth=0.8)
    if shade_deceptive:
        ax.fill_between([xlim[0], 0], 0, ylim[1], color='red', alpha=0.08, label='Deceptive Region')

    # Final styling
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small')

    return fig, ax


# ===== Data Processing Utilities =====

def extract_metrics_from_runs(runs, metric_keys=None, final_only=False):
    """Extract metrics from multiple simulation runs (robust version)."""
    # --- Using the robust version from the first code snippet ---
    if not runs: return {}
    if not isinstance(runs, list) or not all(isinstance(run, dict) for run in runs):
         # Attempt to handle nested structure accidentally passed (e.g., from sweep_results)
         if isinstance(runs, dict) and len(runs) == 1:
             inner_list = list(runs.values())[0]
             if isinstance(inner_list, list) and all(isinstance(r, dict) for r in inner_list):
                 runs = inner_list
             else:
                 print(f"Warning: extract_metrics_from_runs received invalid input type: {type(runs)}. Expected list of dicts.")
                 return {}
         elif isinstance(runs, dict) and len(runs)>1:
              # Maybe it's a dict keyed by param value? Extract the first list.
              first_key = list(runs.keys())[0]
              potential_list = runs[first_key]
              if isinstance(potential_list, list) and all(isinstance(r, dict) for r in potential_list):
                  print(f"Warning: extract_metrics_from_runs received dict. Processing runs for key '{first_key}'.")
                  runs = potential_list
              else:
                  print(f"Warning: extract_metrics_from_runs received invalid input type: {type(runs)}. Expected list of dicts.")
                  return {}
         else:
              print(f"Warning: extract_metrics_from_runs received invalid input type: {type(runs)}. Expected list of dicts.")
              return {}


    first_run = runs[0]
    if metric_keys is None:
        metric_keys = list(first_run.get('history', {}).keys())
        if not metric_keys:
            print("Warning: Could not determine metric keys from run history.")
            return {} # Cannot proceed

    metrics = {}
    for key in metric_keys:
        all_values_for_key = []
        max_len = 0
        valid_run_found = False
        for run in runs:
            history = run.get('history', {})
            if key in history and isinstance(history[key], (list, np.ndarray)):
                 valid_run_found = True
                 if final_only:
                     if len(history[key]) > 0:
                         all_values_for_key.append(history[key][-1])
                     else:
                         all_values_for_key.append(np.nan) # Empty history case
                 else:
                     # Handle potential nested arrays (e.g., cluster_proportions)
                     if isinstance(history[key][0], (list, np.ndarray)):
                         # Stack if possible, otherwise keep as list of arrays/lists
                         try:
                             as_array = np.array(history[key])
                             if as_array.ndim == 2: # Common case like (n_gen, n_clusters)
                                all_values_for_key.append(as_array)
                                max_len = max(max_len, as_array.shape[0])
                             else: # Keep as list of arrays if stacking fails or results in >2D
                                 all_values_for_key.append(history[key])
                                 max_len = max(max_len, len(history[key]))
                         except:
                             all_values_for_key.append(history[key])
                             max_len = max(max_len, len(history[key]))
                     else: # Simple list of scalars
                         all_values_for_key.append(np.asarray(history[key])) # Store as array
                         max_len = max(max_len, len(history[key]))
            else:
                # Append placeholder if key missing or not array-like
                all_values_for_key.append(np.nan if final_only else None)

        if not valid_run_found: # Skip if metric never found
            metrics[key] = np.array([])
            continue

        if final_only:
            metrics[key] = np.array(all_values_for_key)
        else:
            # Pad histories to max_len
            padded_histories = []
            is_nested = any(isinstance(hist, list) and isinstance(hist[0], (list, np.ndarray)) for hist in all_values_for_key if hist is not None and len(hist)>0)

            for hist in all_values_for_key:
                if hist is None: # Run didn't have this metric
                    # Determine padding shape/value based on nested status
                    if is_nested:
                        # Need to know the inner dimension size for padding
                        # Find first valid nested list to infer inner dim
                        inner_dim = 0
                        for h in all_values_for_key:
                            if h is not None and len(h)>0 and isinstance(h[0], (list, np.ndarray)):
                                inner_dim = len(h[0])
                                break
                        if inner_dim > 0:
                             padded_histories.append(np.full((max_len, inner_dim), np.nan))
                        else: # Cannot determine inner shape, pad with NaNs
                             padded_histories.append([np.nan] * max_len)
                    else: # Simple case
                        padded_histories.append(np.full(max_len, np.nan))
                elif is_nested: # Handle padding for list of arrays/lists
                     current_len = len(hist)
                     if current_len < max_len:
                         # Try to pad with last valid row or NaNs
                         if current_len > 0 and isinstance(hist[0], (list, np.ndarray)):
                             inner_dim = len(hist[0])
                             pad_value = np.full(inner_dim, np.nan)
                             padded_hist = list(hist) + [pad_value] * (max_len - current_len)
                         else: # Fallback if cannot determine structure
                             padded_hist = list(hist) + [np.nan] * (max_len - current_len)
                         padded_histories.append(padded_hist)
                     else:
                          padded_histories.append(hist[:max_len]) # Truncate if somehow longer
                else: # Handle padding for simple arrays
                    current_len = len(hist)
                    if current_len < max_len:
                        # Pad with last valid value or NaN
                        pad_value = hist[-1] if current_len > 0 else np.nan
                        padded_histories.append(np.pad(hist, (0, max_len - current_len), 'constant', constant_values=pad_value))
                    else:
                         padded_histories.append(hist[:max_len]) # Truncate if somehow longer

            # Try converting to numpy array, handle potential errors with object dtype
            try:
                 metrics[key] = np.array(padded_histories)
            except ValueError: # Often occurs with inconsistent nested shapes
                 print(f"Warning: Could not convert padded histories for '{key}' to uniform numpy array. Keeping as list of lists/arrays.")
                 metrics[key] = padded_histories # Keep as list if conversion fails


    return metrics


def calculate_metric_statistics(metrics):
    """Calculate statistics (mean, std) for extracted metrics, handling NaNs."""
    # --- Using the robust version from the first code snippet ---
    stats = {}
    for key, values in metrics.items():
        if isinstance(values, list): # Handle list case (e.g., from failed array conversion)
            try: # Try converting again, might work if processing isolated key
                values = np.array(values)
            except ValueError:
                print(f"Warning: Cannot calculate stats for '{key}' as it remains a list of varying structures.")
                stats[f'mean_{key}'] = np.nan # Or handle appropriately
                stats[f'std_{key}'] = np.nan
                continue

        if not isinstance(values, np.ndarray) or values.size == 0:
             stats[f'mean_{key}'] = np.nan
             stats[f'std_{key}'] = np.nan
             continue

        try:
            if values.ndim == 1: # Simple array (final values or 1D history)
                stats[f'mean_{key}'] = np.nanmean(values)
                stats[f'std_{key}'] = np.nanstd(values)
            elif values.ndim == 2: # Multiple runs of history or multiple final values
                stats[f'mean_{key}'] = np.nanmean(values, axis=0)
                stats[f'std_{key}'] = np.nanstd(values, axis=0)
            elif values.ndim == 3: # e.g., Multiple runs of cluster proportions (runs, time, clusters)
                stats[f'mean_{key}'] = np.nanmean(values, axis=0) # Avg over runs -> (time, clusters)
                stats[f'std_{key}'] = np.nanstd(values, axis=0)  # Std over runs -> (time, clusters)
            else: # Higher dimensions? Just take mean/std over the first axis (usually runs)
                 print(f"Warning: Calculating stats for high-dimensional metric '{key}' (ndim={values.ndim}). Averaging over axis 0.")
                 stats[f'mean_{key}'] = np.nanmean(values, axis=0)
                 stats[f'std_{key}'] = np.nanstd(values, axis=0)
        except (TypeError, np.AxisError) as e:
             print(f"Warning: Error calculating stats for '{key}'. Skipping. Error: {e}")
             stats[f'mean_{key}'] = np.nan
             stats[f'std_{key}'] = np.nan

    return stats


def get_cluster_names(params, n_clusters):
    """Extract cluster names from parameters or generate defaults."""
    # --- Using the robust version from the first code snippet ---
    names = []
    # Ensure BELIEF_GENERATION exists and has clusters
    belief_gen_params = params.get('BELIEF_GENERATION', {})
    clusters = belief_gen_params.get('clusters', [])

    for i in range(n_clusters):
        if i < len(clusters) and isinstance(clusters[i], dict) and 'name' in clusters[i]:
            names.append(clusters[i]['name'])
        else:
            names.append(f'Cluster {i}') # Default name if missing
    return names


# ===== Strategy Functions (Simulation Core Logic) =====
# --- Using implementations from the first code snippet (already robust) ---

def generate_bivariate_beliefs(params):
    """Generate beliefs: bivariate normal distribution."""
    n_beliefs_total = params['N_BELIEFS_TOTAL']
    belief_gen_params = params['BELIEF_GENERATION']
    mu_v = belief_gen_params.get('mu_v', 0.0); mu_a = belief_gen_params.get('mu_a', 0.0)
    sigma_v = max(belief_gen_params.get('sigma_v', 1.0), 1e-6)
    sigma_a = max(belief_gen_params.get('sigma_a', 1.0), 1e-6)
    rho = np.clip(belief_gen_params.get('rho', 0.0), -1.0, 1.0)
    mean = [mu_v, mu_a]; cov = [[sigma_v**2, rho*sigma_v*sigma_a], [rho*sigma_v*sigma_a, sigma_a**2]]
    try:
        eigenvalues = np.linalg.eigvalsh(cov)
        if np.any(eigenvalues <= 1e-9): cov += np.eye(2) * (abs(np.min(eigenvalues)) + 1e-6)
        beliefs = np.random.multivariate_normal(mean, cov, n_beliefs_total)
    except np.linalg.LinAlgError:
        print(f"Warning: Bivariate covariance matrix issue. Using diagonal fallback.")
        cov_fallback = [[sigma_v**2, 0], [0, sigma_a**2]]
        beliefs = np.random.multivariate_normal(mean, cov_fallback, n_beliefs_total)
    belief_values = beliefs[:, 0]; belief_alignments = beliefs[:, 1]
    deceptive = np.logical_and(belief_values < 0, belief_alignments > 0)
    initial_deceptive_ratio = np.sum(deceptive) / n_beliefs_total if n_beliefs_total > 0 else 0
    return belief_values, belief_alignments, None, initial_deceptive_ratio

def generate_clustered_beliefs(params):
    """Generate beliefs: mixture of bivariate normal clusters."""
    n_beliefs_total = params['N_BELIEFS_TOTAL']
    belief_gen_params = params['BELIEF_GENERATION']
    cluster_params = belief_gen_params.get('clusters', [])
    if not cluster_params: raise ValueError("No 'clusters' defined for clustered generation.")
    global_correlation = belief_gen_params.get('global_correlation', 0.0)
    belief_values = np.zeros(n_beliefs_total); belief_alignments = np.zeros(n_beliefs_total)
    cluster_indices = np.zeros(n_beliefs_total, dtype=int)
    props = np.array([c.get("prop", 0.0) for c in cluster_params])
    total_prop = np.sum(props)
    if total_prop <= 0: props = np.ones(len(cluster_params)) / len(cluster_params)
    else: props = props / total_prop
    cluster_counts = np.round(props * n_beliefs_total).astype(int)
    diff = n_beliefs_total - np.sum(cluster_counts)
    if diff != 0: cluster_counts[np.argmax(cluster_counts)] += diff
    cluster_counts = np.maximum(0, cluster_counts)
    diff = n_beliefs_total - np.sum(cluster_counts)
    if diff > 0: np.add.at(cluster_counts, np.random.choice(len(cluster_params), size=diff, replace=True), 1)
    belief_index = 0
    for i, cluster in enumerate(cluster_params):
        n_cluster = cluster_counts[i]
        if n_cluster <= 0: continue
        mu_v = cluster.get("mu_v", 0.0); mu_a = cluster.get("mu_a", 0.0)
        sigma_v = max(cluster.get("sigma_v", 1.0), 1e-6)
        sigma_a = max(cluster.get("sigma_a", 1.0), 1e-6)
        correlation = np.clip(cluster.get('rho', global_correlation), -1.0, 1.0)
        mean = [mu_v, mu_a]; cov = [[sigma_v**2, correlation*sigma_v*sigma_a], [correlation*sigma_v*sigma_a, sigma_a**2]]
        try:
            eigenvalues = np.linalg.eigvalsh(cov)
            if np.any(eigenvalues <= 1e-9): cov += np.eye(2) * (abs(np.min(eigenvalues)) + 1e-6)
            beliefs = np.random.multivariate_normal(mean, cov, n_cluster)
        except np.linalg.LinAlgError:
            print(f"Warning: Cluster {i} covariance matrix issue. Using diagonal fallback.")
            cov_fallback = [[sigma_v**2, 0], [0, sigma_a**2]]
            beliefs = np.random.multivariate_normal(mean, cov_fallback, n_cluster)
        end_idx = belief_index + n_cluster
        belief_values[belief_index:end_idx] = beliefs[:, 0]
        belief_alignments[belief_index:end_idx] = beliefs[:, 1]
        cluster_indices[belief_index:end_idx] = i
        belief_index = end_idx
    deceptive = np.logical_and(belief_values < 0, belief_alignments > 0)
    initial_deceptive_ratio = np.sum(deceptive) / n_beliefs_total if n_beliefs_total > 0 else 0
    return belief_values, belief_alignments, cluster_indices, initial_deceptive_ratio

def create_activation_matrix_simple(params):
    """Create activation matrix: simple sparse random."""
    n_beliefs = params['N_BELIEFS_TOTAL']; n_questions = params['N_QUESTIONS']
    act_params = params.get('ACTIVATION_PARAMS', {})
    density = act_params.get('density', params.get('ACTIVATION_DENSITY', 0.1))
    scores = np.zeros((n_beliefs, n_questions))
    n_active = max(1, int(n_questions * density))
    for b in range(n_beliefs):
        if n_questions > 0:
             active_q = np.random.choice(n_questions, size=min(n_active, n_questions), replace=False)
             scores[b, active_q] = 1.0
    return scores

def create_activation_matrix_similarity(params):
    """Create activation matrix: similarity-based."""
    n_beliefs = params['N_BELIEFS_TOTAL']; n_questions = params['N_QUESTIONS']
    act_params = params.get('ACTIVATION_PARAMS', {})
    dim = act_params.get('embedding_dim', 10)
    base_prob = act_params.get('base_prob', 0.2)
    scale = act_params.get('similarity_scale', 5.0)
    noise_std = act_params.get('noise_std', 0.1)
    scores = np.zeros((n_beliefs, n_questions))
    if n_questions == 0: return scores # Handle case with no questions
    q_embed = np.random.normal(0, 1/np.sqrt(dim), (n_questions, dim))
    b_embed = np.random.normal(0, 1/np.sqrt(dim), (n_beliefs, dim))
    for b in range(n_beliefs):
        similarities = np.dot(b_embed[b], q_embed.T)
        for q in range(n_questions):
            if np.random.random() < base_prob:
                strength = 1 / (1 + np.exp(-scale * similarities[q]))
                noise = np.random.normal(0, noise_std)
                scores[b, q] = np.clip(strength + noise, 0, 1)
    return scores

def create_activation_matrix_coverage(params):
    """Create activation matrix with controlled coverage (fraction of beliefs activated by any question)."""
    n_beliefs = params['N_BELIEFS_TOTAL']
    n_questions = params['N_QUESTIONS']
    act_params = params.get('ACTIVATION_PARAMS', {})
    coverage = act_params.get('coverage', 0.5)  # Default 50% coverage
    
    # Initialize scores matrix
    scores = np.zeros((n_beliefs, n_questions))
    
    # Ensure each belief is activated by at least one question (up to coverage)
    n_covered = int(coverage * n_beliefs)
    for b in range(n_covered):
        q = np.random.randint(0, n_questions) if n_questions > 0 else 0
        if n_questions > 0:
            scores[b, q] = 1.0
    
    # Add additional activations to reach desired density
    density = act_params.get('density', 0.1)
    total_activations = int(n_beliefs * n_questions * density)
    current_activations = np.sum(scores)
    remaining = max(0, total_activations - current_activations)
    
    # Add remaining activations randomly
    if n_questions > 0:
        for _ in range(int(remaining)):
            b = np.random.randint(0, n_beliefs)
            q = np.random.randint(0, n_questions)
            scores[b, q] = 1.0
    
    return scores

def create_activation_matrix_correlation(params):
    """Create activation matrix with controlled correlation between question activation patterns."""
    n_beliefs = params['N_BELIEFS_TOTAL']
    n_questions = params['N_QUESTIONS']
    act_params = params.get('ACTIVATION_PARAMS', {})
    correlation = act_params.get('correlation', 0.3)  # Default moderate correlation
    density = act_params.get('density', 0.1)
    
    # Initialize scores matrix
    scores = np.zeros((n_beliefs, n_questions))
    
    if n_questions == 0:
        return scores
    
    # Create correlated question patterns
    # Start with a base pattern and derive others with controlled correlation
    base_pattern = np.random.binomial(1, density, n_beliefs)
    
    for q in range(n_questions):
        # For each question, decide which beliefs to copy from base_pattern
        copy_mask = np.random.binomial(1, correlation, n_beliefs).astype(bool)
        # Generate a random pattern for the rest
        random_pattern = np.random.binomial(1, density, n_beliefs)
        
        # Combine: copy from base where mask is True, use random otherwise
        q_pattern = np.where(copy_mask, base_pattern, random_pattern)
        scores[:, q] = q_pattern
    
    return scores

def initialize_population(params, belief_values=None):
    """Initialize population: random unique beliefs per model."""
    pop_size = params['POPULATION_SIZE']; n_total = params['N_BELIEFS_TOTAL']; n_per = params['N_BELIEFS_PER_MODEL']
    if n_per > n_total: raise ValueError("N_BELIEFS_PER_MODEL > N_BELIEFS_TOTAL")
    pop = np.zeros((pop_size, n_per), dtype=int)
    for m in range(pop_size): pop[m] = np.random.choice(n_total, size=n_per, replace=False)
    return pop

def calculate_question_based_scores(population, belief_values, belief_alignments, activation_scores, params):
    """Calculate model fitness and value based on question activations."""
    pop_size = params['POPULATION_SIZE']; n_q = params['N_QUESTIONS']; eps = 1e-9
    fit_scores = np.zeros(pop_size); val_scores = np.zeros(pop_size)
    for m in range(pop_size):
        b_indices = population[m]
        model_acts = activation_scores[b_indices, :] # Shape: (N_BELIEF_PER_MODEL, N_Q)
        total_act_q = np.sum(model_acts, axis=0) # Shape: (N_Q,)
        act_q_mask = total_act_q > eps
        n_act_q = np.sum(act_q_mask)
        if n_act_q == 0: continue
        act_q_indices = np.where(act_q_mask)[0]
        rel_acts = model_acts[:, act_q_indices] # Shape: (N_BELIEF_PER_MODEL, N_ACT_Q)
        rel_total_act = total_act_q[act_q_indices] # Shape: (N_ACT_Q,)
        weights = rel_acts / (rel_total_act + eps) # Shape: (N_BELIEF_PER_MODEL, N_ACT_Q)
        b_aligns = belief_alignments[b_indices] # Shape: (N_BELIEF_PER_MODEL,)
        b_vals = belief_values[b_indices]     # Shape: (N_BELIEF_PER_MODEL,)
        # Weighted sum: (N_BELIEF, 1) * (N_BELIEF, N_ACT_Q) -> sum over axis 0 -> (N_ACT_Q,)
        q_fitness = np.sum(b_aligns[:, np.newaxis] * weights, axis=0)
        q_value = np.sum(b_vals[:, np.newaxis] * weights, axis=0)
        fit_scores[m] = np.mean(q_fitness) # Average over activated questions
        val_scores[m] = np.mean(q_value)   # Average over activated questions
    return fit_scores, val_scores

def calculate_selection_probs(fitness_scores, beta):
    """Calculate selection probabilities using softmax."""
    max_fit = np.max(fitness_scores); scaled_fit = beta * (fitness_scores - max_fit)
    exp_fit = np.exp(scaled_fit); sum_exp = np.sum(exp_fit)
    if sum_exp == 0: return np.ones_like(fitness_scores) / len(fitness_scores)
    probs = exp_fit / sum_exp; return probs / np.sum(probs) # Normalize again for safety

def reproduce_inheritance(population, parent_indices, params, belief_values=None, belief_alignments=None):
    """Reproduction: Simple inheritance (copy)."""
    return population[parent_indices]

def reproduce_with_mutation(population, parent_indices, params, belief_values, belief_alignments):
    """Reproduction: Inheritance + simple mutation."""
    pop_size = params['POPULATION_SIZE']; n_total = params['N_BELIEFS_TOTAL']; n_per = params['N_BELIEFS_PER_MODEL']
    rate = params.get('MUTATION_RATE', 0.0)
    if rate == 0: return population[parent_indices]
    new_pop = population[parent_indices].copy()
    all_indices = np.arange(n_total)
    for i in range(pop_size):
        child_beliefs = set(new_pop[i])
        for j in range(n_per):
            if np.random.random() < rate:
                available = np.setdiff1d(all_indices, list(child_beliefs), assume_unique=True)
                if len(available) > 0:
                    new_b = np.random.choice(available)
                    old_b = new_pop[i, j]
                    child_beliefs.remove(old_b); child_beliefs.add(new_b)
                    new_pop[i, j] = new_b
    return new_pop

def track_metrics(population, belief_values, belief_alignments, cluster_indices, params):
    """Calculate population metrics for a generation."""
    pop_size=params['POPULATION_SIZE']; n_per=params['N_BELIEFS_PER_MODEL']; total_b = pop_size * n_per
    if total_b == 0: return {'avg_value_raw':np.nan, 'avg_alignment_raw':np.nan, 'std_value_raw':np.nan, 'std_alignment_raw':np.nan, 'deceptive_ratio_raw':np.nan}
    all_b_idx = population.flatten()
    pop_vals = belief_values[all_b_idx]; pop_aligns = belief_alignments[all_b_idx]
    metrics = { 'avg_value_raw': np.mean(pop_vals), 'avg_alignment_raw': np.mean(pop_aligns),
                'std_value_raw': np.std(pop_vals), 'std_alignment_raw': np.std(pop_aligns) }
    deceptive = np.logical_and(pop_vals < 0, pop_aligns > 0)
    metrics['deceptive_ratio_raw'] = np.sum(deceptive) / total_b
    if cluster_indices is not None:
        pop_c_idx = cluster_indices[all_b_idx]
        # Get number of clusters dynamically or from params
        if 'clusters' in params['BELIEF_GENERATION']:
            n_clusters = len(params['BELIEF_GENERATION']['clusters'])
        else:
            n_clusters = (np.max(cluster_indices) + 1) if len(cluster_indices)>0 else 0

        if n_clusters > 0:
            c_counts = np.zeros(n_clusters)
            unique_c, counts = np.unique(pop_c_idx, return_counts=True)
            valid_idx = (unique_c >= 0) & (unique_c < n_clusters) # Ensure indices are valid
            c_counts[unique_c[valid_idx]] = counts[valid_idx]
            metrics['cluster_proportions'] = c_counts / total_b
        else: metrics['cluster_proportions'] = np.array([])
    return metrics


# ===== Core Simulation Function =====

def run_core_simulation(params):
    """Runs the main simulation loop and returns results."""
    # --- Using implementation from the first code snippet ---
    params = copy.deepcopy(params)
    n_generations = params['N_GENERATIONS']; pop_size = params['POPULATION_SIZE']
    beta = params.get('SELECTION_PRESSURE_BETA', 5.0)
    belief_gen_type = params['BELIEF_GENERATION'].get('type', 'bivariate')
    reproduction_type = params.get('REPRODUCTION', 'inheritance')
    activation_type = params.get('ACTIVATION_TYPE', 'simple')
    seed = params.get('SEED', None)
    results_dir = params.get('RESULTS_DIR', '.') # Use RESULTS_DIR from params

    if seed is not None: np.random.seed(seed)

    belief_gen_func = generate_clustered_beliefs if belief_gen_type == 'clustered' else generate_bivariate_beliefs
    activation_func = create_activation_matrix_similarity if activation_type == 'similarity' else create_activation_matrix_simple
    reproduction_func = reproduce_with_mutation if reproduction_type == 'mutation' else reproduce_inheritance
    if reproduction_type == 'mutation' and 'MUTATION_RATE' not in params: params['MUTATION_RATE'] = 0.01

    print(f"Starting run (Seed: {seed}, Dir: {os.path.basename(results_dir)}, Gen: {n_generations}, Pop: {pop_size}, Repro: {reproduction_type}, Act: {activation_type})")
    start_time = time.time()

    beliefs_v, beliefs_a, c_indices, init_deceptive = belief_gen_func(params)
    act_scores = activation_func(params)
    population = initialize_population(params)

    history = { 'fitness': [], 'value': [], 'avg_value_raw': [], 'avg_alignment_raw': [], 'deceptive_ratio_raw': [] }
    if c_indices is not None:
        # Get number of clusters dynamically or from params
        if 'clusters' in params['BELIEF_GENERATION']:
            n_clusters = len(params['BELIEF_GENERATION']['clusters'])
        else:
            n_clusters = (np.max(c_indices) + 1) if len(c_indices)>0 else 0
        if n_clusters > 0: history['cluster_proportions'] = []

    for gen in range(n_generations):
        fit_scores, val_scores = calculate_question_based_scores(population, beliefs_v, beliefs_a, act_scores, params)
        current_metrics = track_metrics(population, beliefs_v, beliefs_a, c_indices, params)

        # Log metrics - use calculated scores for fitness/value, raw metrics for others
        history['fitness'].append(np.mean(fit_scores))
        history['value'].append(np.mean(val_scores))
        history['avg_value_raw'].append(current_metrics.get('avg_value_raw', np.nan))
        history['avg_alignment_raw'].append(current_metrics.get('avg_alignment_raw', np.nan))
        history['deceptive_ratio_raw'].append(current_metrics.get('deceptive_ratio_raw', np.nan))
        if 'cluster_proportions' in history and 'cluster_proportions' in current_metrics:
             history['cluster_proportions'].append(current_metrics['cluster_proportions'])

        if (gen + 1) % max(1, n_generations // 10) == 0:
             print(f"  Gen {gen+1}/{n_generations} - AvgFit: {history['fitness'][-1]:.4f}, AvgVal: {history['value'][-1]:.4f}, Deceptive: {history['deceptive_ratio_raw'][-1]:.3f}")

        sel_probs = calculate_selection_probs(fit_scores, beta)
        parent_idx = np.random.choice(pop_size, size=pop_size, p=sel_probs, replace=True)
        population = reproduction_func(population, parent_idx, params, beliefs_v, beliefs_a)

    # Convert history lists to numpy arrays before returning/saving
    for key in history: history[key] = np.array(history[key])
    print(f"Finished run in {time.time() - start_time:.2f}s. Final AvgFit: {history['fitness'][-1]:.4f}, AvgVal: {history['value'][-1]:.4f}")

    # Compile results, separating arrays for saving
    results = { 'initial_deceptive_ratio': init_deceptive, 'params': params }
    arrays_to_save = {
        'history': history, 'final_population': population,
        'belief_values': beliefs_v, 'belief_alignments': beliefs_a,
        'cluster_indices': c_indices, 'activation_scores': act_scores
    }

    # --- Save results to files (consistent with loading logic) ---
    run_base_name = f"run_{seed if seed is not None else 'nan'}"
    json_path = os.path.join(results_dir, f"{run_base_name}.json")
    npz_path = os.path.join(results_dir, f"{run_base_name}_arrays.npz")

    try:
        # Save non-array data as JSON
        # Need a custom encoder or pre-processing for params if it contains complex types
        # For now, save basic results dict, params might fail if they have numpy types
        serializable_results = {'initial_deceptive_ratio': init_deceptive}
        try:
            with open(json_path, 'w') as f: json.dump(serializable_results, f, indent=2)
        except TypeError:
             print(f"Warning: Basic results not JSON serializable. Saving only arrays.")
             # Fallback: save empty JSON?
             with open(json_path, 'w') as f: json.dump({}, f)


        # Save arrays using np.savez_compressed
        # Need to handle None for cluster_indices
        save_dict = {}
        save_dict['final_population'] = arrays_to_save['final_population']
        save_dict['belief_values'] = arrays_to_save['belief_values']
        save_dict['belief_alignments'] = arrays_to_save['belief_alignments']
        if arrays_to_save['cluster_indices'] is not None:
            save_dict['cluster_indices'] = arrays_to_save['cluster_indices']
        # Don't save activation scores by default to save space, unless needed
        # save_dict['activation_scores'] = arrays_to_save['activation_scores']

        # Handle history dict containing arrays
        history_items = arrays_to_save['history']
        for k, v in history_items.items():
            save_dict[f'history_{k}'] = v # Flatten history keys

        np.savez_compressed(npz_path, **save_dict)
        print(f"Saved run results to {json_path} and {npz_path}")

    except Exception as e:
        print(f"Error saving run results for seed {seed}: {e}")

    # Return the results dictionary with arrays included for in-memory analysis
    results.update(arrays_to_save)
    return results

def run_core_simulation_level4(params):
    """Runs the main simulation loop with Level 4 dynamic features."""
    params = copy.deepcopy(params)
    n_generations = params['N_GENERATIONS']
    pop_size = params['POPULATION_SIZE']
    beta = params.get('SELECTION_PRESSURE_BETA', 5.0)
    belief_gen_type = params['BELIEF_GENERATION'].get('type', 'bivariate')
    reproduction_type = params.get('REPRODUCTION', 'inheritance')
    activation_type = params.get('ACTIVATION_TYPE', 'simple')
    dynamic_features = params.get('DYNAMIC_FEATURES', [])
    seed = params.get('SEED', None)
    results_dir = params.get('RESULTS_DIR', '.')

    if seed is not None:
        np.random.seed(seed)

    # Select appropriate generator functions
    belief_gen_func = generate_clustered_beliefs if belief_gen_type == 'clustered' else generate_bivariate_beliefs
    
    # Level 4: Select activation function based on specified type
    if activation_type == 'similarity':
        activation_func = create_activation_matrix_similarity
    elif activation_type == 'coverage':
        activation_func = create_activation_matrix_coverage
    elif activation_type == 'correlation':
        activation_func = create_activation_matrix_correlation
    else:  # default to simple
        activation_func = create_activation_matrix_simple
    
    reproduction_func = reproduce_with_mutation if reproduction_type == 'mutation' else reproduce_inheritance
    if reproduction_type == 'mutation' and 'MUTATION_RATE' not in params:
        params['MUTATION_RATE'] = 0.01

    print(f"Starting Level 4 run (Seed: {seed}, Dynamic Features: {dynamic_features})")
    start_time = time.time()

    beliefs_v, beliefs_a, c_indices, init_deceptive = belief_gen_func(params)
    act_scores = activation_func(params)
    population = initialize_population(params)

    # Track number of questions for dynamic test
    n_questions_history = [act_scores.shape[1]]

    history = {
        'fitness': [], 'value': [], 'avg_value_raw': [], 'avg_alignment_raw': [],
        'deceptive_ratio_raw': [], 'n_questions': n_questions_history
    }
    if c_indices is not None:
        if 'clusters' in params['BELIEF_GENERATION']:
            n_clusters = len(params['BELIEF_GENERATION']['clusters'])
        else:
            n_clusters = (np.max(c_indices) + 1) if len(c_indices) > 0 else 0
        if n_clusters > 0:
            history['cluster_proportions'] = []

    for gen in range(n_generations):
        # Dynamic features: update test set or alignment scores
        if 'dynamic_test' in dynamic_features:
            act_scores = update_activation_matrix_dynamic(population, beliefs_v, beliefs_a, act_scores, params, gen)
            n_questions_history.append(act_scores.shape[1])
        
        if 'improving_alignment' in dynamic_features:
            beliefs_a = improve_alignment_scores(beliefs_v, beliefs_a, params, gen)
        
        # Run core simulation step
        fit_scores, val_scores = calculate_question_based_scores(population, beliefs_v, beliefs_a, act_scores, params)
        current_metrics = track_metrics(population, beliefs_v, beliefs_a, c_indices, params)

        # Log metrics
        history['fitness'].append(np.mean(fit_scores))
        history['value'].append(np.mean(val_scores))
        history['avg_value_raw'].append(current_metrics.get('avg_value_raw', np.nan))
        history['avg_alignment_raw'].append(current_metrics.get('avg_alignment_raw', np.nan))
        history['deceptive_ratio_raw'].append(current_metrics.get('deceptive_ratio_raw', np.nan))
        if 'cluster_proportions' in history and 'cluster_proportions' in current_metrics:
            history['cluster_proportions'].append(current_metrics['cluster_proportions'])

        if (gen + 1) % max(1, n_generations // 10) == 0:
            print(f"  Gen {gen+1}/{n_generations} - AvgFit: {history['fitness'][-1]:.4f}, AvgVal: {history['value'][-1]:.4f}, Deceptive: {history['deceptive_ratio_raw'][-1]:.3f}, Questions: {act_scores.shape[1]}")

        sel_probs = calculate_selection_probs(fit_scores, beta)
        parent_idx = np.random.choice(pop_size, size=pop_size, p=sel_probs, replace=True)
        population = reproduction_func(population, parent_idx, params, beliefs_v, beliefs_a)

    # Convert history lists to numpy arrays
    for key in history:
        history[key] = np.array(history[key])
    
    print(f"Finished Level 4 run in {time.time() - start_time:.2f}s.")

    # Compile results
    results = {
        'initial_deceptive_ratio': init_deceptive,
        'params': params,
        'history': history,
        'final_population': population,
        'belief_values': beliefs_v,
        'belief_alignments': beliefs_a,
        'cluster_indices': c_indices,
        'activation_scores': act_scores
    }

    # Save results the same way as in run_core_simulation
    run_base_name = f"run_{seed if seed is not None else 'nan'}"
    json_path = os.path.join(results_dir, f"{run_base_name}.json")
    npz_path = os.path.join(results_dir, f"{run_base_name}_arrays.npz")

    try:
        # Save non-array data as JSON
        serializable_results = {'initial_deceptive_ratio': init_deceptive}
        try:
            with open(json_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        except TypeError:
            print(f"Warning: Basic results not JSON serializable. Saving only arrays.")
            with open(json_path, 'w') as f:
                json.dump({}, f)

        # Save arrays using np.savez_compressed
        save_dict = {}
        save_dict['final_population'] = results['final_population']
        save_dict['belief_values'] = results['belief_values']
        save_dict['belief_alignments'] = results['belief_alignments']
        if results['cluster_indices'] is not None:
            save_dict['cluster_indices'] = results['cluster_indices']

        # Handle history dict containing arrays
        history_items = results['history']
        for k, v in history_items.items():
            save_dict[f'history_{k}'] = v  # Flatten history keys

        np.savez_compressed(npz_path, **save_dict)
        print(f"Saved run results to {json_path} and {npz_path}")

    except Exception as e:
        print(f"Error saving run results for seed {seed}: {e}")

    return results


# ===== Configuration Management =====

def load_config(config_path, cmd_args=None):
    """Loads config, applies overrides, prepares results directory."""
    # --- Using the robust version from the first code snippet ---
    default_params = {
        "LEVEL": 1, "N_GENERATIONS": 50, "POPULATION_SIZE": 100, "N_BELIEFS_TOTAL": 1000,
        "N_BELIEFS_PER_MODEL": 50, "N_QUESTIONS": 100, "ACTIVATION_TYPE": "simple",
        "ACTIVATION_PARAMS": {"density": 0.1, "embedding_dim": 10, "base_prob": 0.2, "similarity_scale": 5.0, "noise_std": 0.1},
        "SELECTION_PRESSURE_BETA": 5.0, "N_RUNS": 10,
        "BELIEF_GENERATION": {"type": "bivariate", "mu_v": 0.0, "mu_a": 0.0, "sigma_v": 1.0, "sigma_a": 1.0, "rho": 0.5},
        "REPRODUCTION": "inheritance", "MUTATION_RATE": 0.01, # Default mutation rate if used
        "PARAMETER_SWEEPS": {}, "RUN_2D_SWEEP": False, "2D_SWEEP": {},
    }
    params = default_params.copy()
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'): loaded_params = json.load(f)
            elif config_path.endswith(('.yaml', '.yml')): loaded_params = yaml.safe_load(f)
            else: raise ValueError("Unsupported config format")

        # Simple update (overwrites top level keys) - consider deep merge for nested dicts if needed
        params.update(loaded_params)
        print(f"Loaded configuration from {config_path}")
    except FileNotFoundError: print(f"Warning: Config file not found at {config_path}. Using defaults.")
    except Exception as e: print(f"Error loading config {config_path}: {e}. Using defaults.")

    # Apply command-line overrides
    if cmd_args:
        if cmd_args.level is not None: params['LEVEL'] = cmd_args.level
        if cmd_args.n_runs is not None: params['N_RUNS'] = cmd_args.n_runs
        if cmd_args.beta is not None: params['SELECTION_PRESSURE_BETA'] = cmd_args.beta
        if cmd_args.mutation_rate is not None:
            params['MUTATION_RATE'] = cmd_args.mutation_rate
            if params['MUTATION_RATE'] > 0: params['REPRODUCTION'] = 'mutation'
            # Maybe set to inheritance if rate is 0? Or leave as is.
            elif params['MUTATION_RATE'] == 0 and params['REPRODUCTION'] == 'mutation':
                 print("Warning: mutation_rate=0 specified with REPRODUCTION='mutation'. Reproduction type remains 'mutation' but rate is 0.")
        if cmd_args.rho is not None:
            if params['BELIEF_GENERATION'].get('type') == 'clustered':
                params['BELIEF_GENERATION']['global_correlation'] = cmd_args.rho
            else: # Assume bivariate
                params['BELIEF_GENERATION']['rho'] = cmd_args.rho;
                params['BELIEF_GENERATION']['type'] = 'bivariate' # Ensure type if overriding rho

    # Setup results directory
    level = params.get('LEVEL', 0); timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if cmd_args and cmd_args.output_dir:
        results_dir = cmd_args.output_dir
        if not os.path.isabs(results_dir) and os.path.dirname(results_dir) == '':
            results_dir = os.path.join("results", results_dir)
    else: results_dir = os.path.join("results", f"level{level}_{timestamp}")
    params['RESULTS_DIR'] = results_dir
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # Save effective config
    eff_config_path = os.path.join(results_dir, "effective_config.yaml") # Save as YAML
    try:
        with open(eff_config_path, 'w') as f: yaml.dump(params, f, default_flow_style=False, sort_keys=False)
        print(f"Saved effective configuration to {eff_config_path}")
    except Exception as e: print(f"Warning: Could not save effective config: {e}")
    return params


# ===== Level-Specific Runners / Analysis Orchestration =====

def run_single_parameter_sweep(base_params, param_name, param_values, param_path):
    """Runs parallel simulations for a parameter sweep, saves individual runs."""
    # --- Using the robust version from the first code snippet ---
    print(f"\n=== Running parameter sweep for {param_name} ===")
    start_time = time.time()
    sweep_results_in_memory = {} # Still keep in memory for analysis if needed
    base_results_dir = base_params['RESULTS_DIR']; n_runs = base_params.get('N_RUNS', 1)
    tasks = []
    param_value_map = {} # Map task index back to param value

    for i, value in enumerate(param_values):
        sweep_params = copy.deepcopy(base_params)
        try: # Set parameter value using nested path
            current_dict = sweep_params; path = param_path[:-1]; key = param_path[-1]
            for p in path: current_dict = current_dict[p]
            current_dict[key] = value
        except KeyError as e: print(f"Error setting path {param_path}: {e}. Skipping {param_name}={value}."); continue
        except Exception as e: print(f"Error setting parameter {param_name}={value} at path {param_path}: {e}. Skipping."); continue

        # Create a subdirectory for this specific parameter value's runs
        param_sub_dir = os.path.join(base_results_dir, f"{param_name}_{value}")
        os.makedirs(param_sub_dir, exist_ok=True)
        sweep_params['RESULTS_DIR'] = param_sub_dir # *** CRITICAL: Set subdir for saving ***
        print(f"  Preparing runs for {param_name}={value} (N_RUNS={n_runs}, saving to {param_sub_dir})")
        for run_idx in range(n_runs):
            run_params = copy.deepcopy(sweep_params)
            base_seed = run_params.get('SEED', int(time.time()))
            run_params['SEED'] = base_seed + run_idx * 100 + i # Unique seed
            tasks.append(run_params)
            param_value_map[len(tasks)-1] = value # Track which param value this task belongs to


    num_workers = min(mp.cpu_count(), len(tasks))
    print(f"Starting parallel execution ({len(tasks)} runs, {num_workers} workers)...")
    if not tasks: return {}
    with mp.Pool(processes=num_workers) as pool:
        all_run_results_list = pool.map(run_core_simulation, tasks) # Run and save happens inside

    # Group in-memory results (might consume a lot of memory)
    for task_idx, run_result in enumerate(all_run_results_list):
         param_value = param_value_map.get(task_idx)
         if param_value is not None:
             if param_value not in sweep_results_in_memory: sweep_results_in_memory[param_value] = []
             sweep_results_in_memory[param_value].append(run_result) # Store full result if needed

    print(f"Parameter sweep completed in {time.time() - start_time:.2f} seconds")
    return sweep_results_in_memory # Return in-memory structure if analysis uses it

# ===== NEW: 2D Sweep Functions (from sprawling code) =====

def run_2d_parameter_sweep(params):
    """
    Run a 2D parameter sweep exploring combinations of two parameters.
    (Integrated from sprawling code)

    Args:
        params: Dictionary containing simulation parameters (must include 2D_SWEEP section)

    Returns:
        heatmap_data: Dictionary with 2D sweep results aggregated (means)
    """
    print("\n=== Running 2D parameter sweep ===")
    base_params = copy.deepcopy(params)
    results_dir = base_params['RESULTS_DIR']
    sweep_2d_dir = os.path.join(results_dir, "2d_sweep_analysis") # Specific subdir for 2D results/plots
    os.makedirs(sweep_2d_dir, exist_ok=True)

    # Extract 2D sweep parameters
    sweep_config = base_params.get('2D_SWEEP', {})
    param1_name = sweep_config.get('param1_name')
    param1_values = sweep_config.get('param1_values')
    param1_path = sweep_config.get('param1_path')
    param2_name = sweep_config.get('param2_name')
    param2_values = sweep_config.get('param2_values')
    param2_path = sweep_config.get('param2_path')

    if not all([param1_name, param1_values, param1_path, param2_name, param2_values, param2_path]):
        print("Error: Missing required keys in '2D_SWEEP' configuration.")
        print("Required: param1_name, param1_values, param1_path, param2_name, param2_values, param2_path")
        return None

    # Use specific n_runs for 2D sweep if defined, otherwise use global N_RUNS
    n_runs = sweep_config.get('n_runs', base_params.get('N_RUNS', 3))

    print(f"Sweeping {param1_name} ({param1_values}) vs {param2_name} ({param2_values}) with {n_runs} runs per combo.")

    # Initialize result arrays
    n_param1 = len(param1_values)
    n_param2 = len(param2_values)

    mean_fitness = np.full((n_param1, n_param2), np.nan)
    mean_value = np.full((n_param1, n_param2), np.nan)
    mean_deceptive = np.full((n_param1, n_param2), np.nan) # Use deceptive_ratio_raw

    tasks = []
    param_indices_map = {} # Map task index to (i, j) grid position

    # Prepare tasks for all combinations
    for i, val1 in enumerate(param1_values):
        for j, val2 in enumerate(param2_values):
            combo_params_base = copy.deepcopy(base_params)

            # Create subdirectory for this combo's runs if saving individually
            # combo_run_dir = os.path.join(sweep_2d_dir, f"{param1_name}_{val1}_{param2_name}_{val2}")
            # os.makedirs(combo_run_dir, exist_ok=True)
            # combo_params_base['RESULTS_DIR'] = combo_run_dir # Save individual runs? Maybe too many files.

            # For now, keep results in memory and save aggregated data/plots in sweep_2d_dir

            # Set param1 value
            try:
                current_dict = combo_params_base
                for key in param1_path[:-1]: current_dict = current_dict[key]
                current_dict[param1_path[-1]] = val1
            except Exception as e:
                print(f"Error setting {param1_name}={val1}: {e}. Skipping combo.")
                continue

            # Set param2 value
            try:
                current_dict = combo_params_base
                for key in param2_path[:-1]: current_dict = current_dict[key]
                current_dict[param2_path[-1]] = val2
            except Exception as e:
                print(f"Error setting {param2_name}={val2}: {e}. Skipping combo.")
                continue

            # Prepare runs for this combo
            for run_idx in range(n_runs):
                run_params = copy.deepcopy(combo_params_base)
                # Assign unique seed
                base_seed = run_params.get('SEED', int(time.time()))
                run_params['SEED'] = base_seed + i * 1000 + j * 100 + run_idx # Make seed unique across grid
                tasks.append(run_params)
                param_indices_map[len(tasks)-1] = (i, j) # Store grid index


    # Run simulations in parallel
    if not tasks:
        print("No tasks generated for 2D sweep.")
        return None

    num_workers = min(mp.cpu_count(), len(tasks))
    print(f"Starting parallel execution for 2D sweep ({len(tasks)} total runs)...")
    start_time = time.time()
    with mp.Pool(processes=num_workers) as pool:
        all_run_results = pool.map(run_core_simulation, tasks)
    end_time = time.time()
    print(f"2D parameter sweep runs completed in {end_time - start_time:.2f} seconds")

    # Aggregate results into a dictionary keyed by grid index (i,j)
    combo_results = {} # Key: (i, j), Value: list of run results
    for task_idx, run_result in enumerate(all_run_results):
        indices = param_indices_map.get(task_idx)
        if indices is not None:
            if indices not in combo_results:
                combo_results[indices] = []
            combo_results[indices].append(run_result)

    # Calculate means for each combo and populate the result arrays
    for (i, j), runs in combo_results.items():
        if runs: # Check if there are results for this combo
             # Extract final values for key metrics using robust function
             metrics_final = extract_metrics_from_runs(runs, final_only=True)
             stats_final = calculate_metric_statistics(metrics_final)

             mean_fitness[i, j] = stats_final.get('mean_fitness', np.nan)
             mean_value[i, j] = stats_final.get('mean_value', np.nan)
             mean_deceptive[i, j] = stats_final.get('mean_deceptive_ratio_raw', np.nan)


    # Save raw aggregated results (.npz)
    npz_path = os.path.join(sweep_2d_dir, "2d_sweep_results.npz")
    try:
        np.savez_compressed(
            npz_path,
            param1_name=param1_name, param1_values=np.array(param1_values),
            param2_name=param2_name, param2_values=np.array(param2_values),
            mean_fitness=mean_fitness,
            mean_value=mean_value,
            mean_deceptive=mean_deceptive
        )
        print(f"Saved aggregated 2D sweep results to {npz_path}")
    except Exception as e:
         print(f"Error saving 2D sweep NPZ file: {e}")


    # Create heatmap visualizations using the helper function
    try:
        create_2d_heatmaps(param1_name, param1_values, param2_name, param2_values,
                           mean_fitness, mean_value, mean_deceptive, sweep_2d_dir)
    except Exception as e:
        print(f"Error creating 2D heatmaps: {e}")
        import traceback
        traceback.print_exc()


    # Return data for potential further use (optional)
    heatmap_data = {
        param1_name + '_values': param1_values,
        param2_name + '_values': param2_values,
        'mean_fitness': mean_fitness,
        'mean_value': mean_value,
        'mean_deceptive': mean_deceptive
    }
    return heatmap_data

def create_2d_heatmaps(param1_name, param1_values, param2_name, param2_values,
                       fitness, value, deceptive, results_dir):
    """
    Create heatmap visualizations of 2D parameter sweep results.
    (Integrated from sprawling code)

    Args:
        param1_name, param1_values: Name and values for the first parameter (Y-axis)
        param2_name, param2_values: Name and values for the second parameter (X-axis)
        fitness, value, deceptive: 2D arrays of aggregated results (shape: n_param1 x n_param2)
        results_dir: Directory to save visualization results
    """
    print("Creating 2D heatmap visualizations...")

    # Create figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    if not isinstance(axs, np.ndarray): axs = [axs] # Ensure iterable if only one subplot somehow

    # Format labels for axes
    param1_labels = [f"{v:.2g}" for v in param1_values]
    param2_labels = [f"{v:.2g}" for v in param2_values]

    # Plot 1: Fitness
    try:
        create_heatmap(param2_values, param1_values, fitness.T, # Transpose data for imshow/pcolor
                       xlabel=param2_name, ylabel=param1_name, title='Mean Final Fitness',
                       colorbar_label='Mean Final Fitness', ax=axs[0],
                       xticklabels=param2_labels, yticklabels=param1_labels, annotate=True)
    except Exception as e:
        print(f"Error plotting fitness heatmap: {e}")
        axs[0].set_title('Mean Final Fitness (Error)')


    # Plot 2: True Value
    try:
        # Use a diverging colormap for value, centered around 0
        val_max_abs = np.nanmax(np.abs(value)) if not np.all(np.isnan(value)) else 1.0
        val_min, val_max = -val_max_abs, val_max_abs
        create_heatmap(param2_values, param1_values, value.T, # Transpose data
                       xlabel=param2_name, ylabel=param1_name, title='Mean Final True Value', cmap='RdBu_r',
                       colorbar_label='Mean Final True Value', ax=axs[1], vmin=val_min, vmax=val_max,
                       xticklabels=param2_labels, yticklabels=param1_labels, annotate=True)
    except Exception as e:
        print(f"Error plotting value heatmap: {e}")
        axs[1].set_title('Mean Final True Value (Error)')


    # Plot 3: Deceptive Ratio
    try:
        create_heatmap(param2_values, param1_values, deceptive.T, # Transpose data
                       xlabel=param2_name, ylabel=param1_name, title='Mean Final Deceptive Ratio', cmap='Reds',
                       colorbar_label='Mean Final Deceptive Ratio', ax=axs[2], vmin=0, # Ratio >= 0
                       xticklabels=param2_labels, yticklabels=param1_labels, annotate=True)
    except Exception as e:
        print(f"Error plotting deceptive ratio heatmap: {e}")
        axs[2].set_title('Mean Final Deceptive Ratio (Error)')


    fig.suptitle(f'2D Sweep: {param1_name} vs {param2_name}', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    save_figure(fig, "2d_parameter_heatmaps.png", results_dir)

    # --- Create Value/Fitness Ratio Plot ---
    try:
        fig_ratio, ax_ratio = plt.subplots(figsize=(8, 6))

        # Calculate value/fitness ratio (handle division by zero/nan)
        with np.errstate(divide='ignore', invalid='ignore'): # Suppress warnings
            value_fitness_ratio = value / fitness
        value_fitness_ratio[np.isinf(value_fitness_ratio)] = np.nan # Replace inf with nan
        value_fitness_ratio[np.isnan(fitness) | np.isnan(value)] = np.nan # Propagate NaNs


        # Determine appropriate color limits for the ratio (center around 0 or 1?)
        if not np.all(np.isnan(value_fitness_ratio)):
            ratio_min = np.nanmin(value_fitness_ratio)
            ratio_max = np.nanmax(value_fitness_ratio)
            abs_max = max(abs(ratio_min), abs(ratio_max)) if not np.isnan(ratio_min) else 1.0
            vmin_ratio, vmax_ratio = -abs_max, abs_max # Center diverging map around 0
        else:
            vmin_ratio, vmax_ratio = -1, 1 # Default if all NaN

        cmap_ratio = 'RdBu_r'

        create_heatmap(param2_values, param1_values, value_fitness_ratio.T, # Transpose data
                       xlabel=param2_name, ylabel=param1_name,
                       title='Mean Final (Value / Fitness) Ratio', cmap=cmap_ratio,
                       colorbar_label='Value / Fitness Ratio', ax=ax_ratio,
                       vmin=vmin_ratio, vmax=vmax_ratio,
                       xticklabels=param2_labels, yticklabels=param1_labels, annotate=True)

        fig_ratio.tight_layout()
        save_figure(fig_ratio, "2d_value_fitness_ratio.png", results_dir)
    except Exception as e:
        print(f"Error plotting value/fitness ratio heatmap: {e}")
        plt.close(fig_ratio) # Close figure if error occurs


# ===== Analysis/Visualization Functions (using Loading) =====

def load_and_analyze_sweep(results_dir, param_name):
    """Loads individual run files from sweep subdirs and analyzes."""
    # --- Using implementation from the first code snippet ---
    print(f"\n--- Loading and Analyzing {param_name} sweep from {results_dir} ---")
    analysis_sub_dir = os.path.join(results_dir, f"analysis_{param_name}")
    os.makedirs(analysis_sub_dir, exist_ok=True)

    sweep_subdirs = glob.glob(os.path.join(results_dir, f"{param_name}_*"))
    if not sweep_subdirs:
        print(f"No subdirectories found matching '{param_name}_*'. Cannot analyze sweep.")
        return None

    sweep_results_loaded = {}
    param_values_found = []

    for subdir in sweep_subdirs:
        # Try to parse parameter value from subdir name robustly
        try:
            # Get basename, remove prefix, handle potential non-numeric parts
            basename = os.path.basename(subdir)
            value_str = basename.replace(f"{param_name}_", "")
            param_value = float(value_str) # Attempt conversion
            param_values_found.append(param_value)
        except ValueError:
            print(f"Warning: Could not parse parameter value from directory name: {subdir}. Skipping.")
            continue
        except Exception as e:
             print(f"Error processing directory name {subdir}: {e}. Skipping.")
             continue


        run_files = glob.glob(os.path.join(subdir, "run_*.json"))
        print(f"  Loading {len(run_files)} runs for {param_name}={param_value} from {subdir}...")
        runs_data = []
        for rf in run_files:
            # Load JSON and corresponding NPZ
            json_data = {}
            arrays_loaded = {}
            history = {}
            try:
                with open(rf, 'r') as f: json_data = json.load(f)
            except Exception as e: print(f"    Error loading JSON {rf}: {e}"); continue

            npz_path = rf.replace(".json", "_arrays.npz")
            if os.path.exists(npz_path):
                try:
                    # allow_pickle=True might be needed if history contains complex objects (shouldn't usually)
                    with np.load(npz_path, allow_pickle=True) as npz_data:
                        # Reconstruct history dictionary
                        for key in npz_data.files:
                            if key.startswith('history_'):
                                history[key.replace('history_', '', 1)] = npz_data[key]
                            else:
                                arrays_loaded[key] = npz_data[key]
                        json_data.update(arrays_loaded) # Add arrays back
                        json_data['history'] = history
                        runs_data.append(json_data)
                except Exception as e: print(f"    Error loading NPZ {npz_path}: {e}")
            else:
                print(f"    Warning: NPZ file not found for {rf}. Run data may be incomplete.")
                # Decide if you want to proceed without array data
                # runs_data.append(json_data) # Uncomment to proceed without arrays

        if runs_data: sweep_results_loaded[param_value] = runs_data

    if not sweep_results_loaded:
        print("No valid run results loaded for analysis.")
        return None

    # --- Now perform analysis similar to analyze_parameter_sweep ---
    param_values = sorted(sweep_results_loaded.keys())

    # Determine metric keys from the first loaded run
    if not param_values or not sweep_results_loaded[param_values[0]]:
         print("Error: No valid run data found after loading.")
         return None
    first_run_data = sweep_results_loaded[param_values[0]][0]
    metric_keys = list(first_run_data.get('history', {}).keys())
    if not metric_keys:
        print("Error: Could not determine metric keys from loaded data.")
        return None


    # Calculate final stats
    metric_stats = {}
    for key in metric_keys:
         means = []; stds = []
         for param in param_values:
             runs = sweep_results_loaded.get(param, [])
             if runs:
                 extracted = extract_metrics_from_runs(runs, metric_keys=[key], final_only=True)
                 values = extracted.get(key, np.array([np.nan]))
                 # Check if values array is non-empty before calculating stats
                 if values.size > 0:
                     means.append(np.nanmean(values))
                     stds.append(np.nanstd(values))
                 else:
                     means.append(np.nan); stds.append(np.nan)
             else:
                 means.append(np.nan); stds.append(np.nan)
         metric_stats[f'mean_{key}'] = np.array(means)
         metric_stats[f'std_{key}'] = np.array(stds)

    stats_dict = {'param_values': param_values}; stats_dict.update(metric_stats)

    # --- Generate Plots ---
    metrics_to_plot = ['fitness', 'value', 'deceptive_ratio_raw']
    panels = len(metrics_to_plot)

    # Errorbar plots (Final values)
    fig_err, axs_err = plt.subplots(1, panels, figsize=(7 * panels, 5), sharex=True)
    if panels == 1: axs_err = [axs_err]
    for i, metric in enumerate(metrics_to_plot):
        mean_key=f'mean_{metric}'; std_key=f'std_{metric}'
        if mean_key in stats_dict:
             # Ensure yerr has same shape as y_values
             yerr = stats_dict.get(std_key)
             if yerr is not None and len(yerr) != len(param_values): yerr = None # Mismatch, ignore error bars
             create_errorbar_plot(x_values=param_values, y_values=stats_dict[mean_key], yerr=yerr,
                                 xlabel=f'{param_name}', ylabel=f'Mean Final {metric.replace("_"," ").title()}',
                                 title=f'Final {metric.replace("_"," ").title()} vs {param_name}', ax=axs_err[i])
    fig_err.tight_layout(); save_figure(fig_err, f"sweep_{param_name}_final_metrics.png", analysis_sub_dir)

    # Scatter plot (Final Fitness vs Value)
    if 'mean_fitness' in stats_dict and 'mean_value' in stats_dict:
         annotations = {i: f'{param:.2g}' for i, param in enumerate(param_values)}
         fig_scat, ax_scat = create_scatter_plot(x_values=stats_dict['mean_fitness'], y_values=stats_dict['mean_value'],
                                                xlabel='Mean Final Fitness', ylabel='Mean Final True Value',
                                                title=f'Final Fitness vs. Value ({param_name} sweep)', annotations=annotations,
                                                color=param_values, cmap='viridis', colorbar_label=f'{param_name}')
         # Add error bars to scatter
         if 'std_fitness' in stats_dict and 'std_value' in stats_dict:
             xerr = stats_dict['std_fitness']
             yerr = stats_dict['std_value']
             if len(xerr) != len(param_values): xerr = None
             if len(yerr) != len(param_values): yerr = None
             ax_scat.errorbar(stats_dict['mean_fitness'], stats_dict['mean_value'], xerr=xerr, yerr=yerr,
                         fmt='none', ecolor='gray', alpha=0.5, capsize=3, elinewidth=1)
         fig_scat.tight_layout(); save_figure(fig_scat, f"sweep_{param_name}_fitness_vs_value.png", analysis_sub_dir)

    # History plots for representative runs
    sample_indices = [0]; num_p = len(param_values)
    if num_p > 1: sample_indices.append(num_p // 2)
    if num_p > 2: sample_indices.append(-1)
    sample_params = [param_values[i] for i in sample_indices]
    fig_hist, axs_hist = create_multi_panel_plot(len(sample_params), panels, figsize=(7*panels, 5*len(sample_params)), squeeze=False)
    for i, param in enumerate(sample_params):
        runs = sweep_results_loaded.get(param, [])
        if not runs: continue
        hist_metrics = extract_metrics_from_runs(runs, final_only=False)
        hist_stats = calculate_metric_statistics(hist_metrics)
        for j, metric in enumerate(metrics_to_plot):
            mean_key=f'mean_{metric}'; std_key=f'std_{metric}'
            if mean_key in hist_stats and isinstance(hist_stats[mean_key], np.ndarray):
                 mean_hist = hist_stats[mean_key]
                 std_hist = hist_stats.get(std_key)
                 if mean_hist.ndim == 1: # Check if it's a 1D array (expected)
                     generations = np.arange(len(mean_hist))
                     axs_hist[i, j].plot(generations, mean_hist, label=f'{param_name}={param:.2g}')
                     if std_hist is not None and std_hist.ndim == 1 and len(std_hist) == len(mean_hist):
                          axs_hist[i, j].fill_between(generations, mean_hist - std_hist, mean_hist + std_hist, alpha=0.2)
                     axs_hist[i, j].set_title(f'{metric.replace("_"," ").title()} History ({param_name}={param:.2g})')
                     axs_hist[i, j].set_xlabel('Generation'); axs_hist[i, j].set_ylabel(f'Avg {metric.replace("_"," ").title()}'); axs_hist[i, j].grid(True, alpha=0.3)
                     axs_hist[i, j].legend()
                 else:
                      print(f"Warning: Unexpected shape for mean history of '{metric}' (param={param}). Shape: {mean_hist.shape}")

            # Handle cluster proportions separately if needed (plotting lines for each cluster)
            elif metric == 'cluster_proportions' and mean_key in hist_stats and isinstance(hist_stats[mean_key], np.ndarray):
                 mean_hist = hist_stats[mean_key] # Shape (n_gen, n_clusters)
                 std_hist = hist_stats.get(std_key)
                 if mean_hist.ndim == 2:
                     n_clusters_in_data = mean_hist.shape[1]
                     generations = np.arange(mean_hist.shape[0])
                     # Need cluster names - try to get from first run's params
                     cluster_names = []
                     try:
                         run_params = runs[0].get('params', {})
                         cluster_names = get_cluster_names(run_params, n_clusters_in_data)
                     except:
                         cluster_names = [f'Clust {c}' for c in range(n_clusters_in_data)]

                     cmap_hist = plt.get_cmap('tab10')
                     for c_idx in range(n_clusters_in_data):
                         axs_hist[i, j].plot(generations, mean_hist[:, c_idx], label=cluster_names[c_idx], color=cmap_hist(c_idx))
                         if std_hist is not None and std_hist.ndim == 2 and std_hist.shape == mean_hist.shape:
                             axs_hist[i, j].fill_between(generations, mean_hist[:, c_idx] - std_hist[:, c_idx], mean_hist[:, c_idx] + std_hist[:, c_idx], color=cmap_hist(c_idx), alpha=0.1)
                     axs_hist[i, j].set_title(f'Cluster Proportions ({param_name}={param:.2g})')
                     axs_hist[i, j].set_xlabel('Generation'); axs_hist[i, j].set_ylabel('Proportion'); axs_hist[i, j].grid(True, alpha=0.3)
                     axs_hist[i, j].legend(fontsize='small')
                     axs_hist[i, j].set_ylim(bottom=0)


    fig_hist.tight_layout(); save_figure(fig_hist, f"sweep_{param_name}_avg_histories.png", analysis_sub_dir)

    # Save aggregated stats
    headers = [param_name]; data_cols = [param_values]
    for key in metrics_to_plot:
        mean_k=f'mean_{key}'; std_k=f'std_{key}'
        if mean_k in stats_dict:
            headers.extend([f'mean_final_{key}', f'std_final_{key}'])
            data_cols.extend([stats_dict[mean_k], stats_dict.get(std_k, np.full_like(param_values, np.nan))])
    try:
        np.savetxt(os.path.join(analysis_sub_dir, f"{param_name}_final_stats.csv"), np.column_stack(data_cols),
                   delimiter=",", header=",".join(headers), comments='')
    except Exception as e: print(f"Error saving stats CSV: {e}")

    print(f"Analysis for {param_name} sweep complete. Plots saved to {analysis_sub_dir}")
    return stats_dict

# --- visualize_level3_comparison (modified from sprawling code) ---
def visualize_level3_comparison(scenario_results, analysis_dir):
    """
    Generate visualizations for Level 3 scenario comparisons.
    Works with in-memory scenario results without requiring saved files.

    Args:
        scenario_results: Dict mapping scenario names to lists of run results
        analysis_dir: Directory to save visualization figures
    """
    print("\n--- Generating Level 3 Visualizations ---")
    # No separate figure dir needed if analysis_dir is specific to L3 comparison
    # fig_dir = os.path.join(analysis_dir, "figures")
    # os.makedirs(fig_dir, exist_ok=True)
    fig_dir = analysis_dir # Save directly into the analysis dir

    if not scenario_results:
        print("No scenario results to visualize.")
        return {}

    scenario_names = list(scenario_results.keys())
    print(f"Visualizing scenarios: {', '.join(scenario_names)}")

    # Key metrics for comparison
    metrics_to_compare = ['fitness', 'value', 'deceptive_ratio_raw']
    # Check if cluster data is available in any scenario
    has_clusters = any('cluster_proportions' in run.get('history', {}) for name in scenario_names for run in scenario_results[name])
    if has_clusters:
        metrics_to_compare.append('cluster_proportions') # Add if present

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(scenario_names))]


    # Extract and organize statistics per scenario
    scenario_stats = {}
    for name in scenario_names:
        runs = scenario_results.get(name, [])
        if not runs: continue

        metrics_hist = extract_metrics_from_runs(runs, final_only=False)
        stats_hist = calculate_metric_statistics(metrics_hist)

        metrics_final = extract_metrics_from_runs(runs, final_only=True)
        stats_final = calculate_metric_statistics(metrics_final)

        scenario_stats[name] = {
            'history': stats_hist,
            'final': stats_final,
            'params': runs[0].get('params', {}) # Store params from first run for context
        }

    # --- Create comparison plots ---

    # Figure 1: History comparison for all scenarios
    num_plots = len(metrics_to_compare)
    fig_hist, axs_hist = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5), sharex=True, squeeze=False)
    axs_hist = axs_hist.flatten() # Ensure 1D array

    for i, metric in enumerate(metrics_to_compare):
        ax = axs_hist[i]
        mean_key = f'mean_{metric}'
        std_key = f'std_{metric}'

        if metric == 'cluster_proportions':
             # Special handling for cluster proportions
             cluster_names = []
             n_clusters_plot = 0
             # Find first scenario with cluster data to get names/count
             for name in scenario_names:
                 if name in scenario_stats and mean_key in scenario_stats[name]['history']:
                     n_clusters_plot = scenario_stats[name]['history'][mean_key].shape[1]
                     run_params = scenario_stats[name].get('params',{})
                     cluster_names = get_cluster_names(run_params, n_clusters_plot)
                     break
             if not cluster_names: continue # Skip if no cluster data found

             cmap_cluster = plt.get_cmap('viridis', n_clusters_plot)
             # Plot each cluster proportion as a line - maybe too complex here?
             # Alternative: Plot average proportion of a specific cluster (e.g., deceptive)
             deceptive_cluster_index = -1
             for idx, cname in enumerate(cluster_names):
                 if 'deceptive' in cname.lower():
                     deceptive_cluster_index = idx
                     break

             if deceptive_cluster_index != -1:
                 for j, name in enumerate(scenario_names):
                     if name in scenario_stats and mean_key in scenario_stats[name]['history']:
                          mean_hist = scenario_stats[name]['history'][mean_key] # Shape (gen, cluster)
                          std_hist = scenario_stats[name]['history'].get(std_key)
                          if mean_hist.ndim == 2 and deceptive_cluster_index < mean_hist.shape[1]:
                              generations = np.arange(mean_hist.shape[0])
                              ax.plot(generations, mean_hist[:, deceptive_cluster_index], label=name, color=colors[j % len(colors)])
                              if std_hist is not None and std_hist.ndim == 2 and std_hist.shape == mean_hist.shape:
                                   ax.fill_between(generations,
                                                  mean_hist[:, deceptive_cluster_index] - std_hist[:, deceptive_cluster_index],
                                                  mean_hist[:, deceptive_cluster_index] + std_hist[:, deceptive_cluster_index],
                                                  color=colors[j % len(colors)], alpha=0.15)
                 ax.set_title(f'Avg. Deceptive Cluster Prop.')
                 ax.set_ylabel('Proportion')

             else:
                 ax.set_title('Cluster Proportions (Plot Not Impl.)')


        else: # Standard metric plots
            for j, name in enumerate(scenario_names):
                if name in scenario_stats and mean_key in scenario_stats[name]['history']:
                    stats = scenario_stats[name]['history']
                    mean_hist = stats[mean_key]
                    std_hist = stats.get(std_key)
                    if isinstance(mean_hist, np.ndarray) and mean_hist.ndim == 1:
                        generations = np.arange(len(mean_hist))
                        ax.plot(generations, mean_hist, label=name, color=colors[j % len(colors)])
                        if std_hist is not None and isinstance(std_hist, np.ndarray) and std_hist.shape == mean_hist.shape:
                            ax.fill_between(generations, mean_hist - std_hist, mean_hist + std_hist,
                                            color=colors[j % len(colors)], alpha=0.15)
            ax.set_title(f'{metric.replace("_"," ").title()} History')
            ax.set_ylabel(f'Avg {metric.replace("_"," ").title()}')

        ax.set_xlabel('Generation')
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig_hist.suptitle('Scenario Comparison: Metric Histories')
    fig_hist.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure(fig_hist, "scenario_history_comparison.png", fig_dir)


    # Figure 2: Bar chart of final values for all scenarios
    fig_bar, ax_bar = plt.subplots(figsize=(max(8, 2 * len(metrics_to_compare)), 6))

    num_scenarios = len(scenario_stats)
    bar_width = 0.8 / num_scenarios
    # Use only metrics excluding cluster proportions for simple bar chart
    metrics_for_bar = [m for m in metrics_to_compare if m != 'cluster_proportions']
    x_indices = np.arange(len(metrics_for_bar))

    for j, (name, stats_dict) in enumerate(scenario_stats.items()):
        means = [stats_dict['final'].get(f'mean_{m}', np.nan) for m in metrics_for_bar]
        stds = [stats_dict['final'].get(f'std_{m}', 0) for m in metrics_for_bar]

        pos = x_indices + j * bar_width - (num_scenarios - 1) * bar_width / 2
        color = colors[j % len(colors)]

        ax_bar.bar(pos, means, bar_width, yerr=stds, label=name, color=color, capsize=4)

    ax_bar.set_ylabel('Mean Final Value')
    ax_bar.set_title('Scenario Comparison: Final Metrics')
    ax_bar.set_xticks(x_indices)
    ax_bar.set_xticklabels([m.replace("_"," ").title() for m in metrics_for_bar])
    ax_bar.grid(True, axis='y', alpha=0.3)
    ax_bar.legend()

    fig_bar.tight_layout()
    save_figure(fig_bar, "scenario_final_comparison.png", fig_dir)


    # --- Save Summary Text ---
    summary_text = "Level 3 Scenario Comparison Summary:\n"
    summary_text += "------------------------------------\n"
    for name, stats_dict in scenario_stats.items():
        summary_text += f"\nScenario: {name}\n"
        stats = stats_dict['final']
        for metric in metrics_for_bar: # Summarize only bar metrics
             mean_val = stats.get(f'mean_{metric}', np.nan)
             std_val = stats.get(f'std_{metric}', np.nan)
             summary_text += f"  Final Mean {metric:<18}: {mean_val:.4f} ± {std_val:.4f}\n"

    summary_path = os.path.join(analysis_dir, "level3_comparison_summary.txt")
    try:
        with open(summary_path, "w") as f:
            f.write(summary_text)
        print(f"Saved Level 3 summary to {summary_path}")
    except Exception as e:
        print(f"Error writing Level 3 summary file: {e}")
    print(summary_text)


    print(f"Level 3 visualization complete. Figures saved to {fig_dir}")
    return scenario_stats


def load_and_analyze_level3(results_dir):
    """Loads individual run files from level 3 scenario subdirs and analyzes."""
    # --- Using implementation from the first code snippet ---
    print(f"\n--- Loading and Analyzing Level 3 scenarios from {results_dir} ---")
    analysis_dir = os.path.join(results_dir, "level3_comparison_analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    scenario_subdirs = glob.glob(os.path.join(results_dir, "scenario_*"))
    if not scenario_subdirs:
        print("No scenario subdirectories found. Cannot analyze Level 3.")
        return None

    scenario_results_loaded = {}
    for subdir in scenario_subdirs:
        scenario_name = os.path.basename(subdir).replace("scenario_", "")
        run_files = glob.glob(os.path.join(subdir, "run_*.json"))
        print(f"  Loading {len(run_files)} runs for Scenario: {scenario_name} from {subdir}...")
        runs_data = []
        for rf in run_files:
            json_data = {}; arrays_loaded = {}; history = {}
            try:
                with open(rf, 'r') as f: json_data = json.load(f)
                npz_path = rf.replace(".json", "_arrays.npz")
                if os.path.exists(npz_path):
                    with np.load(npz_path, allow_pickle=True) as npz_data:
                        for key in npz_data.files:
                            if key.startswith('history_'): history[key.replace('history_', '', 1)] = npz_data[key]
                            else: arrays_loaded[key] = npz_data[key]
                        json_data.update(arrays_loaded); json_data['history'] = history
                # Load params from effective config in the scenario subdir if needed
                # This assumes the run_core_simulation didn't save params directly
                if 'params' not in json_data:
                     try:
                         cfg_files = glob.glob(os.path.join(subdir, "effective_config.yaml"))
                         if cfg_files:
                             with open(cfg_files[0], 'r') as cfg_f:
                                 json_data['params'] = yaml.safe_load(cfg_f)
                     except Exception as cfg_e: print(f"    Could not load params from config: {cfg_e}")

                runs_data.append(json_data)
            except Exception as e: print(f"    Error loading run {rf}: {e}")
        if runs_data: scenario_results_loaded[scenario_name] = runs_data

    if not scenario_results_loaded:
        print("No valid Level 3 scenario results loaded.")
        return None

    # --- Now call the L3 visualization function ---
    visualize_level3_comparison(scenario_results_loaded, analysis_dir)
    print(f"Level 3 analysis complete. Plots saved to {analysis_dir}")
    return # Or return calculated stats


# --- Visualization function needed by Level 2 runner ---
def visualize_belief_clusters(params):
    """Visualize initial belief distribution for clustered setup."""
    print("Visualizing initial belief clusters...")
    results_dir = params['RESULTS_DIR']
    vis_params = copy.deepcopy(params)
    vis_params['N_BELIEFS_TOTAL'] = params.get('VIS_N_POINTS', 5000) # Use more points for viz

    try:
        belief_values, belief_alignments, cluster_indices, _ = generate_clustered_beliefs(vis_params)
        cluster_params_list = vis_params['BELIEF_GENERATION'].get('clusters', [])

        fig, ax = create_belief_space_plot(
             belief_values=belief_values,
             belief_alignments=belief_alignments,
             cluster_indices=cluster_indices,
             cluster_params=cluster_params_list,
             title='Initial Belief Space (Clustered)',
             figsize=(9, 7)
        )
        fig.tight_layout()
        save_figure(fig, "initial_belief_space_clusters.png", results_dir)
    except Exception as e:
        print(f"Could not visualize belief clusters: {e}")


# ===== Level Runners =====

# Define run_level0 if missing (simple version)
def run_level0(params):
     """ Basic Level 0 run: High vs Low Rho comparison. """
     print("Running Level 0: Minimal Viable Simulation")
     level0_params = copy.deepcopy(params)
     results_dir = level0_params['RESULTS_DIR']
     print(f"Results will be saved to: {results_dir}")

     # Ensure basic setup
     level0_params['BELIEF_GENERATION'] = { "type": "bivariate", "rho": 0.95 }
     level0_params['REPRODUCTION'] = 'inheritance'
     level0_params['ACTIVATION_TYPE'] = 'simple'

     # High Rho
     high_params = copy.deepcopy(level0_params)
     high_params['SEED'] = 42
     high_result = run_core_simulation(high_params)

     # Low Rho
     low_params = copy.deepcopy(level0_params)
     low_params['BELIEF_GENERATION']['rho'] = 0.1
     low_params['SEED'] = 43
     low_result = run_core_simulation(low_params)

     # --- Simple Plot ---
     fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
     axs[0].plot(high_result['history']['fitness'], label=f'Fitness (rho=0.95)')
     axs[0].plot(high_result['history']['value'], label=f'Value (rho=0.95)', linestyle='--')
     axs[0].plot(low_result['history']['fitness'], label=f'Fitness (rho=0.1)')
     axs[0].plot(low_result['history']['value'], label=f'Value (rho=0.1)', linestyle='--')
     axs[0].set_title('Level 0: Fitness vs Value')
     axs[0].set_ylabel('Score')
     axs[0].grid(True, alpha=0.3)
     axs[0].legend()

     axs[1].plot(high_result['history']['deceptive_ratio_raw'], label=f'Deceptive Ratio (rho=0.95)')
     axs[1].plot(low_result['history']['deceptive_ratio_raw'], label=f'Deceptive Ratio (rho=0.1)')
     axs[1].set_title('Deceptive Belief Ratio (Raw)')
     axs[1].set_xlabel('Generation')
     axs[1].set_ylabel('Ratio')
     axs[1].grid(True, alpha=0.3)
     axs[1].legend()
     axs[1].set_ylim(bottom=0)

     fig.tight_layout()
     save_figure(fig, "level0_comparison.png", results_dir)
     print(f"Level 0 plot saved to {results_dir}")
     return results_dir


def run_level1(params):
    """Run Level 1: Parameter Sweep & Statistical Significance."""
    print("Running Level 1: Parameter Sweep & Statistical Significance")
    level1_params = copy.deepcopy(params); results_dir = level1_params['RESULTS_DIR']
    print(f"Results will be saved to: {results_dir}")

    # --- Run 1D Sweeps ---
    sweep_results = {} # Stores in-memory results if needed
    parameter_sweeps_config = level1_params.get('PARAMETER_SWEEPS', {})
    if not parameter_sweeps_config: print("Warning: No parameter sweeps defined for Level 1.")

    for param_name, sweep_config in parameter_sweeps_config.items():
        param_values = sweep_config.get('values'); param_path = sweep_config.get('param_path')
        if not param_values or not param_path: print(f"Invalid sweep '{param_name}'. Skipping."); continue
        # Run sims & saves individual files AND returns results in memory
        in_memory_results = run_single_parameter_sweep(level1_params, param_name, param_values, param_path)
        sweep_results[param_name] = in_memory_results # Store in-memory results if analysis uses them

        # Analyze by loading saved files
        load_and_analyze_sweep(results_dir, param_name)

    # --- Run 2D Sweep (if configured) ---
    if level1_params.get('RUN_2D_SWEEP', False):
         # This function runs sims, saves aggregated NPZ, and creates heatmaps
         run_2d_parameter_sweep(level1_params)

    print(f"Level 1 finished. Results/Analysis in {results_dir}")
    return results_dir

def run_level2(params):
    """Run Level 2: N-Modal Belief Distributions."""
    print("Running Level 2: N-Modal Belief Distributions")
    level2_params = copy.deepcopy(params); results_dir = level2_params['RESULTS_DIR']
    print(f"Results will be saved to: {results_dir}")

    # Ensure clustered generation
    if level2_params['BELIEF_GENERATION'].get('type') != 'clustered':
        print("Warning: Level 2 requires 'clustered' belief generation. Updating.")
        level2_params['BELIEF_GENERATION']['type'] = 'clustered'
        if 'clusters' not in level2_params['BELIEF_GENERATION'] or not level2_params['BELIEF_GENERATION']['clusters']:
            print("Warning: No clusters defined. Using default 3-cluster setup.")
            level2_params['BELIEF_GENERATION']['clusters'] = [
                {"name": "Benign", "mu_v": 0.8, "mu_a": 0.8, "sigma_v": 0.2, "sigma_a": 0.2, "prop": 0.3},
                {"name": "Neutral", "mu_v": 0.1, "mu_a": 0.7, "sigma_v": 0.2, "sigma_a": 0.2, "prop": 0.4},
                {"name": "Deceptive", "mu_v": -0.6, "mu_a": 0.6, "sigma_v": 0.3, "sigma_a": 0.2, "prop": 0.3}
            ]
    # Visualize initial clusters
    visualize_belief_clusters(level2_params)

    # Run sweeps (same logic as L1, relies on saving individual files, then loading for analysis)
    run_level1(level2_params) # Reuse L1 runner, analysis will load files including cluster data

    print(f"Level 2 finished. Results/Analysis in {results_dir}")
    return results_dir

def run_level3(params):
    """Run Level 3: Comparison of Scenarios."""
    print("Running Level 3: Enhanced Realism Comparison")
    base_params = copy.deepcopy(params); results_dir = base_params['RESULTS_DIR']
    level3_analysis_dir = os.path.join(results_dir, "level3_comparison_analysis") # Dir for final comparison plots
    os.makedirs(level3_analysis_dir, exist_ok=True)
    print(f"Level 3 analysis results will be saved to: {level3_analysis_dir}")
    n_runs = base_params.get('N_RUNS', 5)

    # Define scenarios (ensure mutation rate is handled)
    base_mutation_rate = base_params.get("MUTATION_RATE", 0.01) or 0.01 # Ensure non-zero default if mutation active
    scenarios = {
        "Base": {"REPRODUCTION": "inheritance", "MUTATION_RATE": 0.0, "ACTIVATION_TYPE": "simple"},
        "Mutation": {"REPRODUCTION": "mutation", "MUTATION_RATE": base_mutation_rate, "ACTIVATION_TYPE": "simple"},
        "SimilarityActivation": {"REPRODUCTION": "inheritance", "MUTATION_RATE": 0.0, "ACTIVATION_TYPE": "similarity"},
        "MutationAndSimilarityActivation": {"REPRODUCTION": "mutation", "MUTATION_RATE": base_mutation_rate, "ACTIVATION_TYPE": "similarity"}
    }

    tasks = []
    for scenario_name, mods in scenarios.items(): # Prepare tasks for parallel execution
        scenario_params = copy.deepcopy(base_params); scenario_params.update(mods)
        scenario_run_dir = os.path.join(results_dir, f"scenario_{scenario_name}")
        os.makedirs(scenario_run_dir, exist_ok=True)
        scenario_params['RESULTS_DIR'] = scenario_run_dir # *** Set save dir for each run ***
        print(f"\n  Preparing runs for Scenario: {scenario_name} (N_RUNS={n_runs}, saving to {scenario_run_dir})")
        for run_idx in range(n_runs):
            run_params = copy.deepcopy(scenario_params)
            base_seed = run_params.get('SEED', int(time.time()))
            # Ensure unique seed for each run across scenarios
            run_params['SEED'] = base_seed + run_idx * 100 + hash(scenario_name) % 100
            tasks.append(run_params)

    if not tasks: print("No tasks generated."); return results_dir
    num_workers = min(mp.cpu_count(), len(tasks))
    print(f"\nStarting parallel execution for Level 3 ({len(tasks)} runs, {num_workers} workers)...")
    start_time = time.time()
    with mp.Pool(processes=num_workers) as pool: pool.map(run_core_simulation, tasks) # Run and save happens inside
    print(f"Level 3 runs completed in {time.time() - start_time:.2f}s")

    # --- Analyze by loading saved files ---
    load_and_analyze_level3(results_dir) # This function loads from subdirs and plots to analysis dir

    print(f"Level 3 finished. Results in scenario subdirs, analysis in {level3_analysis_dir}")
    return results_dir

def run_level4(params):
    """Run Level 4: Exploring Test Design & Dynamics."""
    print("Running Level 4: Exploring Test Design & Dynamics")
    base_params = copy.deepcopy(params)
    results_dir = base_params['RESULTS_DIR']
    level4_analysis_dir = os.path.join(results_dir, "level4_comparison_analysis")
    os.makedirs(level4_analysis_dir, exist_ok=True)
    print(f"Level 4 analysis results will be saved to: {level4_analysis_dir}")
    n_runs = base_params.get('N_RUNS', 5)

    # Load scenarios from config if available, otherwise use defaults
    if 'LEVEL4_SCENARIOS' in base_params and base_params['LEVEL4_SCENARIOS']:
        # Use scenarios defined in the config file
        scenarios = base_params['LEVEL4_SCENARIOS']
        print(f"Using {len(scenarios)} scenarios defined in configuration")
    else:
        # Use default scenarios as fallback
        print("No LEVEL4_SCENARIOS found in config, using default scenarios")
        scenarios = {
            "Baseline": {
                "ACTIVATION_TYPE": "simple",
                "DYNAMIC_FEATURES": []
            },
            "High_Coverage": {
                "ACTIVATION_TYPE": "coverage",
                "ACTIVATION_PARAMS": {"coverage": 0.8, "density": 0.1},
                "DYNAMIC_FEATURES": []
            },
            "High_Correlation": {
                "ACTIVATION_TYPE": "correlation",
                "ACTIVATION_PARAMS": {"correlation": 0.7, "density": 0.1},
                "DYNAMIC_FEATURES": []
            },
            "Dynamic_Test": {
                "ACTIVATION_TYPE": "simple",
                "DYNAMIC_FEATURES": ["dynamic_test"],
                "DYNAMIC_ACTIVATION_INTERVAL": 5
            },
            "Improving_Alignment": {
                "ACTIVATION_TYPE": "simple",
                "DYNAMIC_FEATURES": ["improving_alignment"],
                "ALIGNMENT_IMPROVEMENT_INTERVAL": 10
            },
            "Combined_Dynamic": {
                "ACTIVATION_TYPE": "coverage",
                "ACTIVATION_PARAMS": {"coverage": 0.6, "density": 0.1},
                "DYNAMIC_FEATURES": ["dynamic_test", "improving_alignment"],
                "DYNAMIC_ACTIVATION_INTERVAL": 10,
                "ALIGNMENT_IMPROVEMENT_INTERVAL": 15
            }
        }

    tasks = []
    for scenario_name, mods in scenarios.items():
        scenario_params = copy.deepcopy(base_params)
        
        # Apply scenario-specific modifications to parameters
        # Handle nested dictionaries properly
        for key, value in mods.items():
            if isinstance(value, dict) and key in scenario_params and isinstance(scenario_params[key], dict):
                # Deep update for nested dictionaries like ACTIVATION_PARAMS
                scenario_params[key].update(value)
            else:
                # Simple override for top-level parameters
                scenario_params[key] = value
            
        scenario_run_dir = os.path.join(results_dir, f"scenario_{scenario_name}")
        os.makedirs(scenario_run_dir, exist_ok=True)
        scenario_params['RESULTS_DIR'] = scenario_run_dir
        
        print(f"\n  Preparing runs for Scenario: {scenario_name} (N_RUNS={n_runs}, saving to {scenario_run_dir})")
        for run_idx in range(n_runs):
            run_params = copy.deepcopy(scenario_params)
            base_seed = run_params.get('SEED', int(time.time()))
            run_params['SEED'] = base_seed + run_idx * 100 + hash(scenario_name) % 100
            tasks.append(run_params)

    if not tasks:
        print("No tasks generated.")
        return results_dir
        
    num_workers = min(mp.cpu_count(), len(tasks))
    print(f"\nStarting parallel execution for Level 4 ({len(tasks)} runs, {num_workers} workers)...")
    start_time = time.time()
    
    with mp.Pool(processes=num_workers) as pool:
        pool.map(run_core_simulation_level4, tasks)
        
    print(f"Level 4 runs completed in {time.time() - start_time:.2f}s")

    # Load and analyze results
    scenario_results = {}
    for scenario_name in scenarios:
        scenario_dir = os.path.join(results_dir, f"scenario_{scenario_name}")
        run_files = glob.glob(os.path.join(scenario_dir, "run_*.json"))
        
        runs_data = []
        for rf in run_files:
            json_data = {}
            arrays_loaded = {}
            history = {}
            
            try:
                with open(rf, 'r') as f:
                    json_data = json.load(f)
                    
                npz_path = rf.replace(".json", "_arrays.npz")
                if os.path.exists(npz_path):
                    with np.load(npz_path, allow_pickle=True) as npz_data:
                        for key in npz_data.files:
                            if key.startswith('history_'):
                                history[key.replace('history_', '', 1)] = npz_data[key]
                            else:
                                arrays_loaded[key] = npz_data[key]
                        json_data.update(arrays_loaded)
                        json_data['history'] = history
                        
                # Load params from config
                if 'params' not in json_data:
                    cfg_files = glob.glob(os.path.join(scenario_dir, "effective_config.yaml"))
                    if cfg_files:
                        with open(cfg_files[0], 'r') as cfg_f:
                            json_data['params'] = yaml.safe_load(cfg_f)
                    else:
                        # Include the scenario config
                        json_data['params'] = {**base_params, **scenarios[scenario_name]}
                
                runs_data.append(json_data)
                
            except Exception as e:
                print(f"    Error loading run {rf}: {e}")
                
        if runs_data:
            scenario_results[scenario_name] = runs_data

    # Visualize results
    if scenario_results:
        visualize_level4_results(scenario_results, level4_analysis_dir)
    else:
        print("No scenario results loaded for visualization.")

    print(f"Level 4 finished. Results in scenario subdirs, analysis in {level4_analysis_dir}")
    return results_dir

def visualize_level4_results(scenario_results, analysis_dir):
    """Generate visualizations specific to Level 4 results."""
    print("\n--- Generating Level 4 Visualizations ---")
    fig_dir = analysis_dir
    
    if not scenario_results:
        print("No scenario results to visualize.")
        return {}

    scenario_names = list(scenario_results.keys())
    print(f"Visualizing scenarios: {', '.join(scenario_names)}")

    # Normal metrics plus Level 4 specific ones
    metrics_to_compare = ['fitness', 'value', 'deceptive_ratio_raw', 'n_questions']
    
    # Extract and organize statistics per scenario
    scenario_stats = {}
    for name in scenario_names:
        runs = scenario_results.get(name, [])
        if not runs:
            continue

        metrics_hist = extract_metrics_from_runs(runs, final_only=False)
        stats_hist = calculate_metric_statistics(metrics_hist)

        metrics_final = extract_metrics_from_runs(runs, final_only=True)
        stats_final = calculate_metric_statistics(metrics_final)

        scenario_stats[name] = {
            'history': stats_hist,
            'final': stats_final,
            'params': runs[0].get('params', {})
        }

    # --- Create comparison plots ---
    # Question count history for dynamic test scenarios
    if any('dynamic_test' in scenario_stats.get(name, {}).get('params', {}).get('DYNAMIC_FEATURES', []) 
           for name in scenario_names):
        
        fig_q, ax_q = plt.subplots(figsize=(10, 6))
        
        cmap = plt.get_cmap('tab10')
        
        for i, name in enumerate(scenario_names):
            if name in scenario_stats and 'mean_n_questions' in scenario_stats[name]['history']:
                q_history = scenario_stats[name]['history']['mean_n_questions']
                generations = np.arange(len(q_history))
                ax_q.plot(generations, q_history, label=name, color=cmap(i))
                
        ax_q.set_title('Number of Questions Over Time')
        ax_q.set_xlabel('Generation')
        ax_q.set_ylabel('Number of Questions')
        ax_q.grid(True, alpha=0.3)
        ax_q.legend()
        
        fig_q.tight_layout()
        save_figure(fig_q, "question_count_history.png", fig_dir)
    
    # Standard metrics history comparison
    num_plots = len(metrics_to_compare) - 1  # Skip n_questions, handled separately
    fig_hist, axs_hist = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5), squeeze=False)
    axs_hist = axs_hist.flatten()

    for i, metric in enumerate(metrics_to_compare):
        if metric == 'n_questions':
            continue  # Skip, already plotted separately
            
        ax = axs_hist[i]
        mean_key = f'mean_{metric}'
        std_key = f'std_{metric}'

        for j, name in enumerate(scenario_names):
            if name in scenario_stats and mean_key in scenario_stats[name]['history']:
                stats = scenario_stats[name]['history']
                mean_hist = stats[mean_key]
                std_hist = stats.get(std_key)
                if isinstance(mean_hist, np.ndarray) and mean_hist.ndim == 1:
                    generations = np.arange(len(mean_hist))
                    ax.plot(generations, mean_hist, label=name, color=cmap(j % 10))
                    if std_hist is not None and isinstance(std_hist, np.ndarray) and std_hist.shape == mean_hist.shape:
                        ax.fill_between(generations, mean_hist - std_hist, mean_hist + std_hist,
                                      color=cmap(j % 10), alpha=0.15)
        ax.set_title(f'{metric.replace("_"," ").title()} History')
        ax.set_ylabel(f'Avg {metric.replace("_"," ").title()}')
        ax.set_xlabel('Generation')
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig_hist.suptitle('Level 4 Scenario Comparison')
    fig_hist.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure(fig_hist, "level4_history_comparison.png", fig_dir)
    
    # Save summary text
    summary_text = "Level 4 Scenario Comparison Summary:\n"
    summary_text += "------------------------------------\n"
    for name, stats_dict in scenario_stats.items():
        summary_text += f"\nScenario: {name}\n"
        stats = stats_dict['final']
        for metric in metrics_to_compare:
            if metric != 'n_questions':  # Skip questions in summary
                mean_val = stats.get(f'mean_{metric}', np.nan)
                std_val = stats.get(f'std_{metric}', np.nan)
                summary_text += f"  Final Mean {metric:<18}: {mean_val:.4f} ± {std_val:.4f}\n"

    summary_path = os.path.join(analysis_dir, "level4_comparison_summary.txt")
    try:
        with open(summary_path, "w") as f:
            f.write(summary_text)
        print(f"Saved Level 4 summary to {summary_path}")
    except Exception as e:
        print(f"Error writing Level 4 summary file: {e}")
    print(summary_text)

    return scenario_stats


# ===== Main Execution Function =====

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI alignment evolution simulation & analysis")
    parser.add_argument("--config", default="configs/level1_base.yaml", help="Path to config file (YAML or JSON)")
    parser.add_argument("--level", type=int, choices=[0, 1, 2, 3, 4], help="Simulation level (overrides config)")
    parser.add_argument("--output_dir", help="Base directory name for results (in ./results/)")
    parser.add_argument("--n_runs", type=int, help="Override N_RUNS")
    parser.add_argument("--beta", type=float, help="Override SELECTION_PRESSURE_BETA")
    parser.add_argument("--rho", type=float, help="Override correlation rho")
    parser.add_argument("--mutation_rate", type=float, help="Override MUTATION_RATE")
    parser.add_argument("--analyze_only", metavar="RESULTS_DIR", help="Skip simulation, only analyze existing results directory")

    args = parser.parse_args()

    os.makedirs("configs", exist_ok=True); os.makedirs("results", exist_ok=True)

    if args.analyze_only:
        # --- Analysis Only Mode ---
        analysis_target_dir = args.analyze_only
        if not os.path.isdir(analysis_target_dir):
            print(f"Error: Specified analysis directory does not exist: {analysis_target_dir}")
            sys.exit(1)
        print(f"--- Running in Analysis-Only Mode on: {analysis_target_dir} ---")

        # Try to determine level from directory name or config file inside
        level_to_analyze = args.level
        if level_to_analyze is None:
             try:
                 dir_name_lower = os.path.basename(analysis_target_dir).lower()
                 if "level4" in dir_name_lower: level_to_analyze = 4
                 elif "level3" in dir_name_lower: level_to_analyze = 3
                 elif "level2" in dir_name_lower: level_to_analyze = 2
                 elif "level1" in dir_name_lower: level_to_analyze = 1
                 elif "level0" in dir_name_lower: level_to_analyze = 0
                 else:
                     # Try loading config file inside
                     config_file = glob.glob(os.path.join(analysis_target_dir, "effective_config.yaml"))
                     if config_file:
                         with open(config_file[0], 'r') as f: cfg = yaml.safe_load(f)
                         level_to_analyze = cfg.get("LEVEL")
             except Exception as e:
                  print(f"Could not auto-detect level from directory or config: {e}")
             if level_to_analyze is None:
                  print("Could not auto-detect level, please specify with --level.")
                  sys.exit(1)

        print(f"Attempting Level {level_to_analyze} analysis...")
        if level_to_analyze == 0:
             print("Level 0 analysis done during run, check plots in directory.")
             # If Level 0 needs loading, implement here
        elif level_to_analyze == 1 or level_to_analyze == 2:
             print(f"Analyzing Level {level_to_analyze} sweeps...")
             # Identify swept parameters by looking for subdirs like 'param_*'
             subdirs = glob.glob(os.path.join(analysis_target_dir, "*_*"))
             swept_params = set()
             for sd in subdirs:
                 basename = os.path.basename(sd)
                 if '_' in basename and not basename.startswith("analysis_") and not basename.startswith("scenario_") and not basename.startswith("2d_"):
                     # Assume format "paramName_value"
                     param_name = basename.split('_')[0]
                     swept_params.add(param_name)

             if not swept_params:
                 print("Could not identify any swept parameters from subdirectories.")
             else:
                 print(f"Found potential swept parameters: {', '.join(swept_params)}")
                 for param_name in swept_params:
                     load_and_analyze_sweep(analysis_target_dir, param_name)

             # Check for 2D sweep results
             npz_2d = os.path.join(analysis_target_dir, "2d_sweep_analysis", "2d_sweep_results.npz")
             if os.path.exists(npz_2d):
                 print("Found 2D sweep results. Regenerating heatmaps...")
                 try:
                     data = np.load(npz_2d)
                     create_2d_heatmaps(
                         str(data['param1_name']), data['param1_values'],
                         str(data['param2_name']), data['param2_values'],
                         data['mean_fitness'], data['mean_value'], data['mean_deceptive'],
                         os.path.join(analysis_target_dir, "2d_sweep_analysis") # Save plots in its subdir
                     )
                 except Exception as e:
                     print(f"Error regenerating 2D heatmaps: {e}")


        elif level_to_analyze == 3:
             load_and_analyze_level3(analysis_target_dir)
        elif level_to_analyze == 4:
             # For level 4, reuse the scenario loading and visualization
             scenario_results = {}
             scenario_subdirs = glob.glob(os.path.join(analysis_target_dir, "scenario_*"))
             for subdir in scenario_subdirs:
                 scenario_name = os.path.basename(subdir).replace("scenario_", "")
                 run_files = glob.glob(os.path.join(subdir, "run_*.json"))
                 runs_data = []
                 for rf in run_files:
                     try:
                         json_data = {}
                         with open(rf, 'r') as f:
                             json_data = json.load(f)
                         npz_path = rf.replace(".json", "_arrays.npz")
                         if os.path.exists(npz_path):
                             arrays_loaded = {}
                             history = {}
                             with np.load(npz_path, allow_pickle=True) as npz_data:
                                 for key in npz_data.files:
                                     if key.startswith('history_'):
                                         history[key.replace('history_', '', 1)] = npz_data[key]
                                     else:
                                         arrays_loaded[key] = npz_data[key]
                                 json_data.update(arrays_loaded)
                                 json_data['history'] = history
                         runs_data.append(json_data)
                     except Exception as e:
                         print(f"Error loading run {rf}: {e}")
                 if runs_data:
                     scenario_results[scenario_name] = runs_data
             
             if scenario_results:
                 visualize_level4_results(scenario_results, analysis_target_dir)
             else:
                 print("No valid Level 4 scenario results loaded.")
        else: print(f"Analysis for level {level_to_analyze} not implemented in analyze_only mode.")

    else:
        # --- Simulation Mode ---
        config_path = args.config
        # Auto-generate default config if not found
        if not os.path.exists(config_path):
            print(f"Config not found: {config_path}. Generating default.")
            default_cfg = { # Condensed default config
                "LEVEL": 1, "N_GENERATIONS": 50, "POPULATION_SIZE": 100, "N_BELIEFS_TOTAL": 1000,
                "N_BELIEFS_PER_MODEL": 50, "N_QUESTIONS": 100, "ACTIVATION_TYPE": "simple",
                "ACTIVATION_PARAMS": {"density": 0.1, "embedding_dim": 10, "base_prob": 0.2, "similarity_scale": 5.0, "noise_std": 0.1},
                "SELECTION_PRESSURE_BETA": 5.0, "N_RUNS": 10,
                "BELIEF_GENERATION": {"type": "bivariate", "rho": 0.5},
                "REPRODUCTION": "inheritance", "MUTATION_RATE": 0.01,
                "PARAMETER_SWEEPS": {"rho": {"values": [0.1, 0.5, 0.9], "param_path": ["BELIEF_GENERATION", "rho"]}},
                "RUN_2D_SWEEP": False, "2D_SWEEP": {},
                "DYNAMIC_FEATURES": [] # Add for Level 4
            }
            try:
                # Save as YAML (preferred)
                if config_path.endswith(('.yaml', '.yml')):
                     with open(config_path, 'w') as f: yaml.dump(default_cfg, f, sort_keys=False, default_flow_style=False)
                else: # Fallback JSON
                     with open(config_path, 'w') as f: json.dump(default_cfg, f, indent=2)
                print(f"Created default config: {config_path}")
            except Exception as e: print(f"Error creating default config: {e}"); sys.exit(1)


        params = load_config(config_path, args)
        level = params['LEVEL']
        print(f"\n*** Starting Simulation Level {level} ***")
        try: # Run the appropriate level
            if level == 0: run_level0(params)
            elif level == 1: run_level1(params)
            elif level == 2: run_level2(params)
            elif level == 3: run_level3(params)
            elif level == 4: run_level4(params)
            else: print(f"Invalid level: {level}"); sys.exit(1)
            print(f"\n*** Simulation Level {level} Finished ***")
            print(f"Results and analysis saved in: {params['RESULTS_DIR']}")
        except Exception as e: # Catch errors during simulation/analysis
             print(f"\n--- Simulation/Analysis Error ---"); print(f"Error: {e}"); import traceback; traceback.print_exc(); sys.exit(1)

def update_activation_matrix_dynamic(population, belief_values, belief_alignments, activation_scores, params, gen):
    """Update activation matrix over generations by adding new questions targeting common beliefs."""
    if gen % params.get('DYNAMIC_ACTIVATION_INTERVAL', 10) != 0:
        return activation_scores  # Only update at specific intervals
    
    n_beliefs = params['N_BELIEFS_TOTAL']
    n_questions = params['N_QUESTIONS']
    
    # Skip if we don't have room to add new questions
    if activation_scores.shape[1] >= n_questions * 2:  # Limit the growth
        return activation_scores
    
    # Identify common beliefs in the population, especially high-fitness ones
    pop_size = params['POPULATION_SIZE']
    all_beliefs = population.flatten()
    belief_counts = np.bincount(all_beliefs, minlength=n_beliefs)
    
    # Create a new question that targets common beliefs, especially deceptive ones
    deceptive = np.logical_and(belief_values < 0, belief_alignments > 0)
    target_beliefs = np.logical_and(belief_counts > 0, deceptive)
    
    if np.sum(target_beliefs) > 0:
        # Add a new question
        new_question = np.zeros((n_beliefs, 1))
        new_question[target_beliefs, 0] = 1.0
        
        # Concatenate to activation matrix
        activation_scores = np.hstack((activation_scores, new_question))
        
        print(f"  Gen {gen}: Added new question targeting {np.sum(target_beliefs)} common deceptive beliefs")
    
    return activation_scores

def improve_alignment_scores(belief_values, belief_alignments, params, gen):
    """Simulate evaluators getting better at assigning alignment scores over time."""
    if gen % params.get('ALIGNMENT_IMPROVEMENT_INTERVAL', 20) != 0 or gen == 0:
        return belief_alignments  # Only update at specific intervals
    
    # Gradually increase correlation between value and alignment
    belief_gen_params = params['BELIEF_GENERATION']
    current_rho = belief_gen_params.get('improving_rho', belief_gen_params.get('rho', belief_gen_params.get('global_correlation', 0.0)))
    target_rho = belief_gen_params.get('target_rho', 0.9)
    improvement_rate = belief_gen_params.get('improvement_rate', 0.1)
    
    # Calculate new rho, moving toward target
    new_rho = current_rho + (target_rho - current_rho) * improvement_rate
    new_rho = min(new_rho, target_rho)  # Cap at target
    belief_gen_params['improving_rho'] = new_rho
    
    # Recalculate alignment scores based on new correlation
    # This creates a weighted blend of old alignments and true values
    weight = (new_rho - current_rho) / (1 - current_rho) if current_rho < 1 else 0
    updated_alignments = (1 - weight) * belief_alignments + weight * belief_values
    
    print(f"  Gen {gen}: Improved alignment correlation from {current_rho:.3f} to {new_rho:.3f}")
    return updated_alignments