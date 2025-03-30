# -*- coding: utf-8 -*-
"""
Visualization Script for "Modelling the evolution of alignment and values"

Generates publication-quality figures from simulation results across
different levels (L1 Sweeps, L2 Clusters, L3 Scenarios, L4 Dynamics).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import glob
import yaml # For loading config
import shutil # For copying existing plot
import warnings # To suppress specific warnings if needed

# --- Configuration ---

# 1. Base directory containing the L1, L2, L3, L4 result folders
RESULTS_BASE_DIR = "./results" # Adjust if your top-level results folder is different

# 2. Specific Directory Names (MUST MATCH YOUR ACTUAL FOLDER NAMES)
L1_DIR_NAME = "level1_final_results"
L2_DIR_NAME = "level2_final_results"
L3_SCENARIO_DIR_NAMES = { # Keep keys consistent with PLOT_STYLE, values are FOLDER names
    "Base": "level3_final_results_base",
    "Mutation": "level3_final_results_mutation",
    "Similarity": "level3_final_results_similarity",
    "Both": "level3_final_results_both",
}

L4_SCENARIO_DIR_NAMES = { # Map Scenario Name used in plots to FOLDER name
    "Baseline": "scenario_Baseline",                 # CORRECTED folder name
    "Mutation": "scenario_Mutation",                 # CORRECTED folder name
    "Coverage_Mid": "scenario_Coverage_Mid",         # CORRECTED folder name
    "Correlation_Mid": "scenario_Correlation_Mid",     # CORRECTED folder name
    "Dynamic_Test": "scenario_Dynamic_Test",         # CORRECTED folder name
    "Improving_Align": "scenario_Improving_Align",     # CORRECTED folder name
    "Combined_Dynamic_Mutation": "scenario_Combined_Dynamic_Mutation", # CORRECTED folder name
}

# 3. Analysis subdirs and CSV/NPZ filenames (used for loading aggregated data)
ANALYSIS_FILES = {
    "rho": {"subdir": "analysis_rho", "csv": "rho_final_stats.csv", "param_col": "rho"},
    "beta": {"subdir": "analysis_beta", "csv": "beta_final_stats.csv", "param_col": "beta"},
    "N_Questions": {"subdir": "analysis_N_Questions", "csv": "N_Questions_final_stats.csv", "param_col": "N_Questions"},
    "deceptive_prop": {"subdir": "analysis_deceptive_prop", "csv": "deceptive_prop_final_stats.csv", "param_col": "deceptive_prop"},
    "global_correlation": {"subdir": "analysis_global_correlation", "csv": "global_correlation_final_stats.csv", "param_col": "global_correlation"},
    "2d_sweep": {"subdir": "2d_sweep_analysis", "npz": "2d_sweep_results.npz"}
}

# 4. Metrics to plot & their readable names (UPDATED LABELS)
METRICS_INFO = {
    'fitness': {'label': 'Alignment Fitness', 'mean_col': 'mean_final_fitness', 'std_col': 'std_final_fitness'},
    'value': {'label': 'True Value', 'mean_col': 'mean_final_value', 'std_col': 'std_final_value'},
    'deceptive_ratio_raw': {'label': 'Deceptive Belief Ratio', 'mean_col': 'mean_final_deceptive_ratio_raw', 'std_col': 'std_final_deceptive_ratio_raw'}
    # Cluster metrics will be added dynamically if detected
}
HISTORY_METRICS_L4 = ['fitness', 'value', 'deceptive_ratio_raw', 'n_questions'] # Include n_questions for L4 history plots

# 5. Output directory for figures
FIGURE_OUTPUT_DIR = "paper_figures"
FIGURE_DPI = 300
SAVE_PDF = True # Also save vector PDF format

# 6. Plotting style customization
plt.style.use('seaborn-v0_8-darkgrid') # Consider 'seaborn-v0_8-paper' or 'seaborn-v0_8-ticks'
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'figure.titlesize': 16, 'figure.dpi': FIGURE_DPI,
    'lines.linewidth': 2.0, 'lines.markersize': 6,
    'errorbar.capsize': 3,
    'font.weight': 'bold',  # Add bold font weight
    'axes.labelweight': 'bold',  # Bold axis labels
    'axes.titleweight': 'bold',  # Bold titles
})

# Use distinct colors/markers/lines for scenario comparisons (up to 10)
SCENARIO_CMAP = plt.colormaps['tab10'] # Updated cmap access
SCENARIO_MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', '+', '.']
SCENARIO_LINES = ['-', '--', ':', '-.',
                  (0, (1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)),
                  (0, (1, 5)), (0, (5, 5))]

def get_plot_style(names_list):
    """Generate consistent plot styles for a list of scenario names."""
    styles = {}
    num_names = len(names_list)
    # Updated cmap access for robustness
    cmap_func = plt.colormaps.get_cmap('tab10')
    colors_to_use = cmap_func.colors if hasattr(cmap_func, 'colors') else [cmap_func(i/max(1, num_names-1)) for i in range(num_names)]

    for i, name in enumerate(names_list):
         styles[name] = {'color': colors_to_use[i % len(colors_to_use)],
                         'marker': SCENARIO_MARKERS[i % len(SCENARIO_MARKERS)],
                         'linestyle': SCENARIO_LINES[i % len(SCENARIO_LINES)]}
    styles['default'] = {'color': colors_to_use[0], 'marker': 'o', 'linestyle': '-'}
    return styles

# Example: Define styles based on L3 and L4 names union if comparing across levels,
# or generate dynamically when needed. Let's generate dynamically.
PLOT_STYLE_L3 = get_plot_style(list(L3_SCENARIO_DIR_NAMES.keys()))
PLOT_STYLE_L4 = get_plot_style(list(L4_SCENARIO_DIR_NAMES.keys()))

ERROR_ALPHA = 0.15

# --- Helper Functions ---

def save_figure(fig, filename, results_dir=FIGURE_OUTPUT_DIR, dpi=FIGURE_DPI, bbox_inches='tight'):
    """Save a matplotlib figure with consistent settings."""
    os.makedirs(results_dir, exist_ok=True)
    base_filepath = os.path.join(results_dir, filename)
    try:
        fig.savefig(f"{base_filepath}.png", dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved figure: {base_filepath}.png")
        if SAVE_PDF:
             fig.savefig(f"{base_filepath}.pdf", bbox_inches=bbox_inches)
             print(f"Saved figure: {base_filepath}.pdf")
    except Exception as e:
        print(f"Error saving figure {base_filepath}: {e}")
    finally:
        plt.close(fig)

def load_sweep_data(data_sources, analysis_info_key):
    """Loads aggregated CSV data for all sources for a specific sweep."""
    sweep_info = ANALYSIS_FILES.get(analysis_info_key)
    if not sweep_info or 'csv' not in sweep_info:
        print(f"Error: Analysis info for '{analysis_info_key}' not found or missing CSV info.")
        return {}

    data_dict = {}
    print(f"\n--- Loading aggregated CSV data for '{sweep_info['subdir']}/{sweep_info['csv']}' ---")
    for source_name, source_dir in data_sources.items():
        if not os.path.isdir(source_dir): # Check if source dir exists
            print(f"  Warning: Source directory not found for {source_name} at {source_dir}")
            continue
        # Path for aggregated CSVs usually within an analysis subdir of the source dir
        csv_path = os.path.join(source_dir, sweep_info['subdir'], sweep_info['csv'])
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                param_col = sweep_info['param_col']
                if param_col not in df.columns:
                    # Try finding the parameter column case-insensitively or with underscores
                    potential_cols = [c for c in df.columns if c.lower() == param_col.lower() or c.lower().replace('_','') == param_col.lower().replace('_','')]
                    if potential_cols:
                         param_col = potential_cols[0]
                         print(f"  Adjusted param col to '{param_col}'")
                    else:
                        raise ValueError(f"Parameter column '{sweep_info['param_col']}' not found in {csv_path}")

                df[param_col] = pd.to_numeric(df[param_col])
                df = df.sort_values(by=param_col).reset_index(drop=True)

                # Dynamically add cluster info to METRICS_INFO if detected
                for i in range(10): # Check for up to 10 clusters
                     # Assume format like 'mean_final_cluster_proportions_0'
                     mean_col_name = f'mean_final_cluster_proportions_{i}'
                     std_col_name = f'std_final_cluster_proportions_{i}'
                     if mean_col_name in df.columns:
                         metric_key = f'cluster_prop_{i}'
                         if metric_key not in METRICS_INFO:
                             # Need cluster names later, store placeholders now
                             METRICS_INFO[metric_key] = {'label': f'Clust. {i} Prop.', 'mean_col': mean_col_name, 'std_col': std_col_name, 'is_cluster': True}
                             print(f"  Detected cluster metric: {metric_key}")
                     else:
                         # Stop checking if we don't find the next sequential column
                         break

                data_dict[source_name] = df
                print(f"  Loaded aggregated data for: {source_name} ({len(df)} rows)")
            except Exception as e:
                print(f"  Error loading or processing {csv_path}: {e}")
        else:
            print(f"  Warning: Aggregated CSV file not found for {source_name} at {csv_path}")
    return data_dict


# NEW function to load history data from individual NPZ files for L4 scenarios
def load_scenario_history_data(scenario_dirs):
    """Loads history arrays from NPZ files for L4 scenarios and computes stats."""
    scenario_stats = {}
    print("\n--- Loading history data for L4 scenarios from NPZ files ---")

    for name, scenario_dir in scenario_dirs.items():
        if not os.path.isdir(scenario_dir):
            print(f"  Warning: Directory not found for scenario '{name}': {scenario_dir}")
            continue

        run_files = glob.glob(os.path.join(scenario_dir, "run_*_arrays.npz"))
        if not run_files:
            print(f"  Warning: No run NPZ files found for scenario '{name}' in {scenario_dir}")
            continue

        print(f"  Processing {len(run_files)} runs for scenario: {name}...")
        histories_per_metric = {} # {metric_name: [run1_hist_array, run2_hist_array, ...]}
        max_len = 0

        for rf in run_files:
            try:
                with np.load(rf, allow_pickle=True) as data:
                    # Check which history metrics are available in this file
                    available_metrics = [k.replace('history_', '') for k in data.files if k.startswith('history_')]
                    if not available_metrics: continue

                    current_max_len = 0
                    for metric in available_metrics:
                        hist_key = f'history_{metric}'
                        if hist_key not in data: continue # Skip if key somehow missing
                        hist_array = data[hist_key]
                        # Ensure hist_array is numpy array and has length
                        if not isinstance(hist_array, np.ndarray) or hist_array.size == 0: continue

                        if metric not in histories_per_metric:
                            histories_per_metric[metric] = []
                        histories_per_metric[metric].append(hist_array)
                        current_max_len = max(current_max_len, len(hist_array))
                    max_len = max(max_len, current_max_len)

            except Exception as e:
                print(f"    Error loading {rf}: {e}")

        # Compute stats (mean/std history) for this scenario
        if histories_per_metric and max_len > 0:
            stats = {}
            for metric, run_hists in histories_per_metric.items():
                 # Pad histories to max_len for consistent stats calculation
                 padded_hists = []
                 # Determine expected shape based on first valid history item for this metric
                 first_valid_hist = next((h for h in run_hists if h is not None and h.size > 0), None)
                 if first_valid_hist is None: continue # Skip metric if no valid data
                 inner_shape = first_valid_hist.shape[1:] if first_valid_hist.ndim > 1 else ()

                 for hist in run_hists:
                     current_len = len(hist)
                     if current_len == max_len:
                         padded_hists.append(hist)
                     elif current_len < max_len:
                          # Pad with NaN
                          pad_shape = (max_len - current_len,) + inner_shape
                          pad_block = np.full(pad_shape, np.nan)
                          try:
                               padded = np.concatenate((hist, pad_block), axis=0)
                               padded_hists.append(padded)
                          except ValueError as pad_e: # Handle shape mismatch during concat if inner shape changed?
                                print(f"      Warning: Concatenation failed for {metric} (len {current_len} vs {max_len}, shape {hist.shape}): {pad_e}. Appending NaNs.")
                                nan_shape = (max_len,) + inner_shape
                                padded_hists.append(np.full(nan_shape, np.nan))

                     else: # hist longer than max_len? Should not happen if max_len correct. Truncate.
                          padded_hists.append(hist[:max_len])

                 try:
                     # Ensure all lists/arrays in padded_hists have the same shape before stacking
                     ref_shape = (max_len,) + inner_shape
                     valid_hists_for_stacking = [h for h in padded_hists if hasattr(h, 'shape') and h.shape == ref_shape]
                     if not valid_hists_for_stacking:
                          print(f"    Warning: No valid histories with shape {ref_shape} found for metric '{metric}' in scenario '{name}'. Skipping stats.")
                          continue

                     stacked_hists = np.array(valid_hists_for_stacking)
                     with warnings.catch_warnings(): # Suppress RuntimeWarning for mean of empty slice
                          warnings.simplefilter("ignore", category=RuntimeWarning)
                          stats[metric] = {
                               'mean': np.nanmean(stacked_hists, axis=0),
                               'std': np.nanstd(stacked_hists, axis=0)
                          }
                 except ValueError as ve:
                     print(f"    Warning: Could not stack histories for metric '{metric}' in scenario '{name}' (likely inconsistent shapes despite padding: {ve}). Skipping stats.")
                 except Exception as calc_e:
                     print(f"    Error calculating stats for metric '{metric}' in scenario '{name}': {calc_e}")

            scenario_stats[name] = stats
        else:
            print(f"  No valid history data processed for scenario '{name}'.")

    return scenario_stats


def load_2d_sweep_data(level_dir):
    """Loads the 2D sweep NPZ file."""
    sweep_info = ANALYSIS_FILES.get("2d_sweep")
    if not sweep_info or 'npz' not in sweep_info:
        print("Error: 2D sweep info missing in ANALYSIS_FILES.")
        return None
    npz_path = os.path.join(level_dir, sweep_info['subdir'], sweep_info['npz'])
    print(f"\n--- Loading 2D data from '{npz_path}' ---")
    if os.path.exists(npz_path):
        try:
            data = np.load(npz_path)
            print("  Loaded 2D sweep data.")
            p1_name = str(data['param1_name']) if 'param1_name' in data else 'param1'
            p2_name = str(data['param2_name']) if 'param2_name' in data else 'param2'
            # Check for required keys
            required_keys = ['param1_values', 'param2_values', 'mean_fitness', 'mean_value', 'mean_deceptive']
            if not all(key in data for key in required_keys):
                 print(f"  Error: Missing one or more required keys in {npz_path}: {required_keys}")
                 return None
            return {
                'param1_name': p1_name, 'param1_values': data['param1_values'],
                'param2_name': p2_name, 'param2_values': data['param2_values'],
                'mean_fitness': data['mean_fitness'],
                'mean_value': data['mean_value'],
                'mean_deceptive': data['mean_deceptive']
            }
        except Exception as e:
            print(f"  Error loading {npz_path}: {e}")
            return None
    else:
        print(f"  Warning: NPZ file not found at {npz_path}")
        return None

def get_config_params(config_dir):
    """Loads effective config parameters from YAML or JSON."""
    params = None
    # Preferentially load from the specific run dir, fallback to parent dir
    paths_to_check = [
        os.path.join(config_dir, "effective_config.yaml"),
        os.path.join(config_dir, "effective_config.json"),
        # Check parent dir too, handles case where config is in main level dir
        # but we are looking inside a scenario subdir
        os.path.join(os.path.dirname(config_dir), "effective_config.yaml"),
        os.path.join(os.path.dirname(config_dir), "effective_config.json")
    ]
    # Also check the base results dir for a global config? Less likely needed.
    # paths_to_check.append(os.path.join(RESULTS_BASE_DIR, "effective_config.yaml"))

    load_path = None
    for p in paths_to_check:
        if os.path.exists(p):
            load_path = p
            break

    if load_path:
        try:
            with open(load_path, 'r') as f:
                params = yaml.safe_load(f) if load_path.endswith(".yaml") else json.load(f)
            # print(f"  Loaded config from {load_path}") # Verbose
        except Exception as e:
            print(f"  Warning: Could not load config from {load_path}: {e}")
    else:
        print(f"  Warning: Config file not found in {config_dir} or its parent.")
    return params

def get_cluster_names_from_params(params):
    """Extracts cluster names from a loaded params dictionary."""
    if not params: return [] # Return empty list if no params
    clusters = params.get('BELIEF_GENERATION', {}).get('clusters', [])
    names = []
    if clusters:
        names = [c.get('name', f'Cluster {i}') for i, c in enumerate(clusters)]
        # print(f"  Extracted cluster names: {names}") # Verbose
    return names

def find_first_npz(directory):
    """Finds the first run_*_arrays.npz file in a directory or its immediate subdirs."""
    # Check current directory first
    npz_files = sorted(glob.glob(os.path.join(directory, "run_*_arrays.npz")))
    if npz_files:
        return npz_files[0]

    # Check one level deeper (e.g., inside sweep param subdir like beta_5.0)
    sub_npz_files = sorted(glob.glob(os.path.join(directory, "*", "run_*_arrays.npz")))
    if sub_npz_files:
        return sub_npz_files[0]

    print(f"  Warning: No run_*_arrays.npz files found in {directory} or its immediate subdirs")
    return None

# Utility function to create belief space plots (copied from main script, minor tweaks)
def create_belief_space_plot(belief_values, belief_alignments, cluster_indices=None,
                             cluster_params=None, colors=None, alpha=0.6,
                             xlabel='True Value (v)', ylabel='Alignment Signal (a)', # Updated label
                             title='Belief Space Distribution',
                             figsize=(7, 6), highlight_indices=None, # Default size adjusted
                             highlight_marker='X', highlight_size=100, # Increased highlight size
                             highlight_alpha=0.9, shade_deceptive=True,
                             xlim=None, ylim=None, ax=None):
    """Create a plot of the belief space, optionally on existing axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if colors is None:
        # Use tab10 colormap by default
        cmap = plt.colormaps.get_cmap('tab10')
        colors = cmap.colors if hasattr(cmap, 'colors') else [cmap(i/9) for i in range(10)]

    # Ensure data are numpy arrays
    belief_values = np.asarray(belief_values)
    belief_alignments = np.asarray(belief_alignments)
    if cluster_indices is not None:
        cluster_indices = np.asarray(cluster_indices)

    # Check for empty data
    if belief_values.size == 0 or belief_alignments.size == 0:
        print(f"Warning: Empty belief data provided for plot '{title}'. Skipping.")
        ax.text(0.5, 0.5, 'No Belief Data', ha='center', va='center', transform=ax.transAxes, color='red', fontsize=14, fontweight='bold')
        ax.set_title(title)
        return fig, ax

    # Auto-determine limits if not provided, handle potential NaNs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning) # Ignore NaN warnings
        val_min, val_max = np.nanmin(belief_values), np.nanmax(belief_values)
        align_min, align_max = np.nanmin(belief_alignments), np.nanmax(belief_alignments)

    # Check if min/max are NaN (happens if all values are NaN)
    if np.isnan(val_min) or np.isnan(val_max) or np.isnan(align_min) or np.isnan(align_max):
         print(f"Warning: Belief values/alignments contain only NaNs for plot '{title}'. Skipping.")
         ax.text(0.5, 0.5, 'Invalid Belief Data (NaNs)', ha='center', va='center', transform=ax.transAxes, color='red', fontsize=14, fontweight='bold')
         ax.set_title(title)
         return fig, ax # Return early

    if xlim is None:
        pad_x = max(1.0, (val_max - val_min) * 0.1) if not np.isclose(val_max, val_min) else 1.0
        xlim = (val_min - pad_x, val_max + pad_x)
    if ylim is None:
        pad_y = max(1.0, (align_max - align_min) * 0.1) if not np.isclose(align_max, align_min) else 1.0
        ylim = (align_min - pad_y, align_max + pad_y)

    # Plot by cluster or all points
    if cluster_indices is not None and len(cluster_indices) == len(belief_values): # Ensure indices match data
        has_cluster_params = isinstance(cluster_params, list) and len(cluster_params) > 0
        valid_mask = ~np.isnan(cluster_indices)
        if not np.any(valid_mask): n_clusters = 0
        else: n_clusters = int(np.nanmax(cluster_indices[valid_mask])) + 1

        for i in range(n_clusters):
            cluster_mask = valid_mask & (cluster_indices == i)
            if np.sum(cluster_mask) == 0: continue # Skip empty clusters
            cluster_name = cluster_params[i].get('name', f'Cluster {i}') if has_cluster_params and i < len(cluster_params) else f'Cluster {i}'
            ax.scatter(belief_values[cluster_mask], belief_alignments[cluster_mask],
                       alpha=alpha, color=colors[i % len(colors)], label=cluster_name, s=20)
        if has_cluster_params:
            for i, cluster in enumerate(cluster_params):
                 ax.scatter([cluster.get('mu_v', 0)], [cluster.get('mu_a', 0)], color=colors[i % len(colors)],
                           marker='X', s=180, linewidth=2, edgecolors='black',
                           label=f'{cluster.get("name", f"Cluster {i}")} Mean', zorder=10)
    else:
        ax.scatter(belief_values, belief_alignments, alpha=alpha, color=colors[0], label='Beliefs', s=20)

    # Highlight points
    if highlight_indices is not None and len(highlight_indices) > 0:
        highlight_indices = np.asarray(highlight_indices)
        valid_indices_mask = (highlight_indices >= 0) & (highlight_indices < len(belief_values))
        valid_indices = highlight_indices[valid_indices_mask].astype(int)
        if len(valid_indices) > 0:
             ax.scatter(belief_values[valid_indices], belief_alignments[valid_indices], alpha=highlight_alpha,
                       color='orange', marker=highlight_marker, s=highlight_size, edgecolors='black',
                       linewidth=0.5, label='Highlighted', zorder=5)

    # Lines and shading
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.6, linewidth=1.5)
    if shade_deceptive:
        ax.fill_between([xlim[0], 0], 0, ylim[1], color='red', alpha=0.12, label='Deceptive Region (v<0, a>0)', zorder=0)

    # Final styling
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, fontweight='bold'); 
    ax.set_ylabel(ylabel, fontweight='bold'); 
    ax.set_title(title, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.7) # Lighter grid
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # Remove duplicate labels
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize='medium', framealpha=0.9)

    return fig, ax

# Utility function to create heatmaps (copied from main script, refined)
def create_heatmap(x_values, y_values, z_values, # x=cols, y=rows, z=(rows, cols)
                   xlabel=None, ylabel=None, title=None,
                   cmap='viridis', figsize=(8, 6.5), # Default size adjusted
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
         return fig, ax # Return early

    # Handle cases where all data is NaN
    all_nan = np.all(np.isnan(z_values))
    if all_nan:
        print(f"Warning: All Z-values are NaN for heatmap '{title}'. Displaying empty plot.")
        vmin_eff, vmax_eff = 0, 1 # Set dummy vmin/vmax for colormap display
    elif vmin is None and vmax is None: # Auto-range if not specified
        vmin_eff = np.nanmin(z_values)
        vmax_eff = np.nanmax(z_values)
        # Handle case where min == max
        if np.isclose(vmin_eff, vmax_eff):
             vmin_eff -= 0.5
             vmax_eff += 0.5
    else: # Use provided vmin/vmax
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

    x_ticks = np.arange(z_values.shape[1]) # Columns correspond to X
    y_ticks = np.arange(z_values.shape[0]) # Rows correspond to Y
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Use provided tick labels or format values
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontweight='bold') # Rotated and bolded
    elif x_values is not None and len(x_values) == z_values.shape[1]:
         ax.set_xticklabels([f"{v:.2g}" for v in x_values], rotation=45, ha="right", fontweight='bold')
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels, fontweight='bold')
    elif y_values is not None and len(y_values) == z_values.shape[0]:
        ax.set_yticklabels([f"{v:.2g}" for v in y_values], fontweight='bold')

    if annotate and not all_nan: # Only annotate if data exists
        for i in range(z_values.shape[0]): # Iterate rows (Y)
            for j in range(z_values.shape[1]): # Iterate columns (X)
                val = z_values[i, j]
                if not np.isnan(val):
                     # Use imshow norm object for correct color calculation
                     norm_val = heatmap.norm(val)
                     bgcolor = heatmap.cmap(norm_val)
                     # Calculate luminance (YIQ formula)
                     luminance = 0.299*bgcolor[0] + 0.587*bgcolor[1] + 0.114*bgcolor[2]
                     textcolor = 'white' if luminance < 0.5 else 'black'
                     # Use larger font size for annotations with bold text
                     ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=textcolor,
                             fontsize=11, fontweight='bold')

    return fig, ax


# --- Publication Quality Plotting Functions (Refined Labels/Titles) ---

def plot_1d_sweep(data_df, swept_param, metrics_info, output_filename_base, title_prefix=""):
    """Generates publication-quality plots for a single 1D sweep."""
    print(f"\nGenerating 1D Sweep plots for {title_prefix}{swept_param}...")
    metrics_to_plot_keys = [k for k in metrics_info if not metrics_info[k].get('is_cluster', False)]
    num_metrics = len(metrics_to_plot_keys)
    if num_metrics == 0:
         print("  No standard metrics found to plot.")
         return

    style = PLOT_STYLE_L3.get('default', {}) # Use a default style

    # --- Figure 1: Metrics vs Parameter (Line Plot) ---
    fig1, axs1 = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4), sharex=True, squeeze=False)
    axs1 = axs1.flatten()
    param_name_nice = swept_param.replace("_", " ").title() # Nicer name for labels/titles

    for i, metric_key in enumerate(metrics_to_plot_keys):
        ax = axs1[i]
        info = metrics_info[metric_key]
        mean_col, std_col = info.get('mean_col'), info.get('std_col')

        if mean_col and std_col and mean_col in data_df.columns and swept_param in data_df.columns:
            x_values = data_df[swept_param]
            y_values = data_df[mean_col]
            y_err = data_df[std_col] if std_col in data_df.columns else None

            ax.plot(x_values, y_values,
                    color=style.get('color'), marker=style.get('marker'),
                    linestyle=style.get('linestyle'))
            if y_err is not None:
                y_err_clean = np.nan_to_num(y_err.values) # Replace NaN errors with 0
                ax.fill_between(x_values, y_values - y_err_clean, y_values + y_err_clean,
                                color=style.get('color'), alpha=ERROR_ALPHA, linewidth=0)

            ax.set_xlabel(param_name_nice)
            ax.set_ylabel(f"Mean Final {info['label']}") # Use refined label
            ax.grid(True, linestyle=':', alpha=0.7) # Lighter grid
            if 'Deceptive' in info['label']: # Check refined label
                # Set sensible Y limits for ratios (slightly below 0, slightly above max or 1)
                valid_y = y_values.dropna()
                upper_limit = max(1.05, valid_y.max() * 1.1) if not valid_y.empty else 1.05
                ax.set_ylim(bottom=-0.02, top=upper_limit)
            if 'True Value' in info['label']: # Check refined label
                ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.7)
        else:
             # Provide more context in missing data message
             ax.set_title(f"{info['label']} (Data Missing)")
             ax.text(0.5, 0.5, f"Data Missing\n({mean_col} or\n{swept_param})",
                     ha='center', va='center', transform=ax.transAxes, color='red')

    fig1.suptitle(f"{title_prefix}Final Metrics vs {param_name_nice}", fontsize=plt.rcParams['figure.titlesize'])
    fig1.tight_layout(rect=[0, 0.03, 1, 0.94])
    save_figure(fig1, f"{output_filename_base}_vs_param")

    # --- Figure 2: Fitness vs Value Scatter ---
    fv_info = metrics_info.get('fitness')
    vv_info = metrics_info.get('value')
    if fv_info and vv_info and fv_info['mean_col'] in data_df.columns and vv_info['mean_col'] in data_df.columns:
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
        x_values = data_df[fv_info['mean_col']]
        y_values = data_df[vv_info['mean_col']]
        x_err = data_df[fv_info['std_col']] if fv_info['std_col'] in data_df.columns else None
        y_err = data_df[vv_info['std_col']] if vv_info['std_col'] in data_df.columns else None
        colors = data_df[swept_param]
        param_name_label = swept_param.replace("_", " ").title()

        # Handle case where all colors might be the same (results in TypeError for cbar)
        try:
             scatter = ax2.scatter(x_values, y_values, c=colors, cmap='viridis', alpha=0.9, s=50, zorder=10)
             cbar = fig2.colorbar(scatter, ax=ax2)
             cbar.set_label(f"Swept Parameter ({param_name_label})") # Clearer colorbar label
        except TypeError: # Likely happens if 'colors' has only one unique value
             scatter = ax2.scatter(x_values, y_values, alpha=0.9, s=50, zorder=10)
             print("  Scatter plot: Only one unique value for color, omitting colorbar.")


        if x_err is not None and y_err is not None:
            x_err_clean = np.nan_to_num(x_err.values)
            y_err_clean = np.nan_to_num(y_err.values)
            ax2.errorbar(x_values, y_values, xerr=x_err_clean, yerr=y_err_clean,
                         fmt='none', ecolor='grey', alpha=0.5, capsize=0, elinewidth=1, zorder=5)

        # Annotate points clearly
        annotations = {i: f'{val:.2g}' for i, val in enumerate(data_df[swept_param])}
        for idx, text in annotations.items():
            # Check bounds using .iloc for safety with pandas indexing
            if idx < len(x_values) and idx < len(y_values):
                 ax2.annotate(text, (x_values.iloc[idx], y_values.iloc[idx]),
                             textcoords="offset points", xytext=(4, 4), ha='left', fontsize=8, alpha=0.9)

        ax2.set_xlabel(f"Mean Final {fv_info['label']}") # Use refined label
        ax2.set_ylabel(f"Mean Final {vv_info['label']}") # Use refined label
        ax2.set_title(f"{title_prefix}Alignment Fitness vs True Value") # Refined title
        ax2.grid(True, linestyle=':', alpha=0.7) # Lighter grid
        ax2.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.7)
        ax2.axvline(0, color='black', linestyle=':', linewidth=1, alpha=0.7)
        fig2.tight_layout()
        save_figure(fig2, f"{output_filename_base}_fitness_vs_value")


def plot_2d_heatmaps(data_2d, output_filename_base, title_prefix=""):
    """Generates publication-quality heatmap plots for 2D sweep results."""
    print(f"\nGenerating 2D Heatmap plots for {title_prefix}...")
    if data_2d is None:
        print("  No 2D data loaded, skipping heatmaps.")
        return

    p1_name = data_2d['param1_name'].replace("_", " ").title()
    p1_vals = data_2d['param1_values']
    p2_name = data_2d['param2_name'].replace("_", " ").title()
    p2_vals = data_2d['param2_values']
    mean_fitness = data_2d.get('mean_fitness')
    mean_value = data_2d.get('mean_value')
    mean_deceptive = data_2d.get('mean_deceptive')

    if mean_fitness is None or mean_value is None or mean_deceptive is None:
         print("  Error: Missing required data arrays (fitness, value, deceptive) in 2D data.")
         return

    # Change layout from horizontal to vertical (1x3 to 3x1)
    fig1, axs1 = plt.subplots(3, 1, figsize=(8, 17)) # Changed from (17, 5) to (8, 17)
    p1_labels = [f"{v:.2g}" for v in p1_vals]
    p2_labels = [f"{v:.2g}" for v in p2_vals]

    # Use refined labels from METRICS_INFO
    fitness_label = METRICS_INFO.get('fitness', {}).get('label', 'Fitness')
    value_label = METRICS_INFO.get('value', {}).get('label', 'True Value')
    decep_label = METRICS_INFO.get('deceptive_ratio_raw', {}).get('label', 'Deceptive Ratio')

    # Ensure data shapes match expected (n_p1 x n_p2) for heatmaps
    expected_shape = (len(p1_vals), len(p2_vals))
    if mean_fitness.shape != expected_shape or mean_value.shape != expected_shape or mean_deceptive.shape != expected_shape:
        print(f"  Error: Data shape mismatch in 2D data. Expected {expected_shape}, got "
              f"Fit:{mean_fitness.shape}, Val:{mean_value.shape}, Dec:{mean_deceptive.shape}")
        plt.close(fig1)
        return

    create_heatmap(p2_vals, p1_vals, mean_fitness, ax=axs1[0], annotate=True,
                   xlabel=p2_name, ylabel=p1_name, title=f'Final {fitness_label}',
                   cmap='viridis', colorbar_label=fitness_label,
                   xticklabels=p2_labels, yticklabels=p1_labels)
    # Calculate robust vmin/vmax for RdBu_r centered potentially around 0
    val_data_clean = mean_value[~np.isnan(mean_value)]
    val_max_abs = np.max(np.abs(val_data_clean)) if val_data_clean.size > 0 else 1.0
    vmin_val, vmax_val = -val_max_abs, val_max_abs
    create_heatmap(p2_vals, p1_vals, mean_value, ax=axs1[1], annotate=True,
                   xlabel=p2_name, ylabel=p1_name, title=f'Final {value_label}',
                   cmap='RdBu_r', vmin=vmin_val, vmax=vmax_val, colorbar_label=value_label,
                   xticklabels=p2_labels, yticklabels=p1_labels)
    # Calculate robust vmax for deceptive ratio (vmin is 0)
    decep_data_clean = mean_deceptive[~np.isnan(mean_deceptive)]
    vmax_dec = np.max(decep_data_clean) if decep_data_clean.size > 0 else 1.0
    create_heatmap(p2_vals, p1_vals, mean_deceptive, ax=axs1[2], annotate=True,
                   xlabel=p2_name, ylabel=p1_name, title=f'Final {decep_label}',
                   cmap='Reds', vmin=0, vmax=max(0.01, vmax_dec), colorbar_label=decep_label,
                   xticklabels=p2_labels, yticklabels=p1_labels)

    fig1.suptitle(f'{title_prefix}2D Sweep: {p1_name} vs {p2_name}', fontsize=plt.rcParams['figure.titlesize'])
    fig1.tight_layout(rect=[0, 0.01, 1, 0.98]) # Adjusted rect for vertical layout
    save_figure(fig1, f"{output_filename_base}_main_heatmaps")

def plot_l2_cluster_composition(data_df, swept_param, cluster_names, output_filename_base):
    """Generates publication-quality stacked bar plot for Level 2 final cluster composition."""
    print(f"\nGenerating L2 Cluster Composition plot for {swept_param} sweep...")
    if not cluster_names:
        print("  No cluster names provided, cannot generate composition plot.")
        return
    n_clusters = len(cluster_names)
    param_name_nice = swept_param.replace("_", " ").title()

    # Find columns dynamically based on the known pattern
    cluster_mean_cols = sorted([col for col in data_df.columns if col.startswith('mean_final_cluster_prop_')])
    actual_n_clusters = len(cluster_mean_cols)

    if actual_n_clusters == 0:
        print(f"  Warning: No cluster proportion columns found in data for {swept_param} sweep.")
        return
    if actual_n_clusters != n_clusters:
         print(f"  Warning: Found {actual_n_clusters} cluster proportion columns, but expected {n_clusters} based on config. Plotting found columns.")
         # Adjust cluster_names if possible, otherwise labels might mismatch
         if actual_n_clusters < n_clusters:
             cluster_names = cluster_names[:actual_n_clusters]
         # else: might need to generate default names for extras

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(data_df)), 5))
    param_values = data_df[swept_param]
    x_pos = np.arange(len(param_values))
    bottom = np.zeros(len(param_values))
    cmap_func = plt.colormaps.get_cmap('viridis', actual_n_clusters) # Use a distinct colormap

    for i in range(actual_n_clusters):
        mean_col = cluster_mean_cols[i]
        proportions = data_df[mean_col].fillna(0).values
        label = cluster_names[i] if i < len(cluster_names) else f"Cluster {i}"
        ax.bar(x_pos, proportions, bottom=bottom, label=label, color=cmap_func(i), width=0.7)
        bottom += proportions

    ax.set_xlabel(param_name_nice)
    ax.set_ylabel('Mean Final Proportion')
    ax.set_title(f'Mean Final Cluster Composition vs {param_name_nice}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{val:.2g}" for val in param_values])
    ax.set_ylim(0, max(1.05, np.max(bottom)*1.05 if len(bottom)>0 else 1.05)) # Ensure y-limit is at least 1.05
    ax.grid(True, axis='y', linestyle=':', alpha=0.7) # Lighter grid
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), title="Belief Clusters", fontsize='small')
    fig.tight_layout(rect=[0, 0.01, 0.85, 0.95]) # Adjust rect to make space for legend
    save_figure(fig, f"{output_filename_base}_cluster_composition")


def plot_sweep_comparison_lines(data_dict, swept_param, metrics_info, output_filename, plot_styles):
    """Creates publication-quality multi-panel line plot comparing scenarios."""
    metrics_to_plot_keys = [k for k in metrics_info if not metrics_info[k].get('is_cluster', False)]
    num_metrics = len(metrics_to_plot_keys)
    if num_metrics == 0 or not data_dict:
         print(f"Warning: No data or metrics to plot for {output_filename}")
         return

    fig, axs = plt.subplots(1, num_metrics, figsize=(5.5 * num_metrics, 4.5), sharex=True, squeeze=False)
    axs = axs.flatten()
    param_name_nice = swept_param.replace("_", " ").title()

    print(f"\nGenerating Comparison Line Plot: {output_filename}")

    for i, metric_key in enumerate(metrics_to_plot_keys):
        ax = axs[i]
        info = metrics_info[metric_key]
        mean_col, std_col = info.get('mean_col'), info.get('std_col')

        for scenario_name, df in data_dict.items():
            # Check if necessary columns exist in this scenario's dataframe
            if mean_col and std_col and mean_col in df.columns and swept_param in df.columns:
                style = plot_styles.get(scenario_name, plot_styles['default'])
                x_values = df[swept_param]
                y_values = df[mean_col]
                y_err = df[std_col] if std_col in df.columns else None

                ax.plot(x_values, y_values,
                        label=scenario_name,
                        color=style.get('color'), marker=style.get('marker'),
                        linestyle=style.get('linestyle'),
                        linewidth=plt.rcParams['lines.linewidth'],
                        markersize=plt.rcParams['lines.markersize']+2)

                if y_err is not None:
                     y_err_clean = np.nan_to_num(y_err.values)
                     ax.fill_between(x_values, y_values - y_err_clean, y_values + y_err_clean,
                                     color=style.get('color'), alpha=ERROR_ALPHA, linewidth=0)
            else:
                 print(f"  Skipping plot for {scenario_name} on metric {metric_key} - data missing.")


        ax.set_xlabel(param_name_nice, fontweight='bold')
        ax.set_ylabel(f"Final {info['label']}", fontweight='bold') # Removed redundant "Mean"
        ax.grid(True, linestyle=':', alpha=0.7) # Lighter grid
        if i == num_metrics - 1: # Place legend only on the last plot
             # Adjust legend position to avoid overlap
             ax.legend(title="Scenario", loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='medium', title_fontsize='medium')
        if 'Deceptive' in info['label']: ax.set_ylim(bottom=-0.02) # Adjust y-lim for ratios
        if 'True Value' in info['label']: ax.axhline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.7)

    fig.suptitle(f"Scenario Comparison vs {param_name_nice}", fontsize=plt.rcParams['figure.titlesize'], fontweight='bold')
    # Adjust tight_layout based on whether legend is present
    right_margin = 0.88 if num_metrics > 0 else 1.0 # Leave space for legend if plots exist
    fig.tight_layout(rect=[0, 0.03, right_margin, 0.93])
    save_figure(fig, output_filename)


def plot_sweep_comparison_bars(data_dict, swept_param, metrics_info, output_filename, plot_styles):
    """Creates publication-quality multi-panel grouped bar plot comparing scenarios."""
    metrics_to_plot_keys = [k for k in metrics_info if not metrics_info[k].get('is_cluster', False)]
    num_metrics = len(metrics_to_plot_keys)
    if num_metrics == 0 or not data_dict:
        print(f"Warning: No data or metrics to plot for {output_filename}")
        return

    fig, axs = plt.subplots(1, num_metrics, figsize=(5.5 * num_metrics, 4.5), sharey=False, squeeze=False)
    axs = axs.flatten()
    param_name_nice = swept_param.replace("_", " ").title()

    print(f"\nGenerating Comparison Bar Plot: {output_filename}")

    scenario_names = list(data_dict.keys())
    num_scenarios = len(scenario_names)
    if num_scenarios == 0: return

    # Find common parameter values across all scenarios for consistent x-axis
    all_param_values = set()
    for df in data_dict.values():
         if swept_param in df.columns:
             all_param_values.update(df[swept_param].unique())
    param_values = sorted(list(all_param_values))
    x_indices = np.arange(len(param_values))

    total_bar_width = 0.8
    bar_width = total_bar_width / num_scenarios

    for i, metric_key in enumerate(metrics_to_plot_keys):
        ax = axs[i]
        info = metrics_info[metric_key]
        mean_col, std_col = info.get('mean_col'), info.get('std_col')

        for j, scenario_name in enumerate(scenario_names):
            df = data_dict.get(scenario_name)
            if df is None or mean_col not in df.columns or swept_param not in df.columns:
                print(f"  Skipping bars for {scenario_name} on metric {metric_key} - data missing.")
                continue

            # Reindex to ensure bars align correctly for missing param values
            df_reindexed = df.set_index(swept_param).reindex(param_values)
            means = df_reindexed[mean_col].fillna(np.nan).values # Use NaN for missing data
            stds = df_reindexed[std_col].fillna(0).values if std_col in df.columns else np.zeros_like(means)

            pos_offset = (j - (num_scenarios - 1) / 2) * bar_width
            bar_positions = x_indices + pos_offset
            style = plot_styles.get(scenario_name, {})

            ax.bar(bar_positions, means, bar_width, yerr=stds,
                   label=scenario_name, color=style.get('color'),
                   capsize=plt.rcParams['errorbar.capsize'], error_kw={'alpha': 0.7, 'lw': 1.5})

        ax.set_xlabel(param_name_nice, fontweight='bold')
        ax.set_ylabel(f"Final {info['label']}", fontweight='bold') # Removed redundant "Mean"
        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"{val:.2g}" for val in param_values], fontweight='bold')
        ax.grid(True, axis='y', linestyle=':', alpha=0.7) # Lighter grid
        if i == num_metrics - 1: # Place legend only on the last plot
             ax.legend(title="Scenario", loc='center left', bbox_to_anchor=(1.05, 0.5), 
                     fontsize='medium', title_fontsize='medium')
        if 'Deceptive' in info['label']: ax.set_ylim(bottom=0) # Ratios start at 0
        if 'True Value' in info['label']: ax.axhline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
        # Add title to each subplot
        ax.set_title(f"{info['label']}", fontweight='bold')

    fig.suptitle(f"Scenario Comparison vs {param_name_nice}", fontsize=plt.rcParams['figure.titlesize'], fontweight='bold')
    # Adjust tight_layout based on whether legend is present
    right_margin = 0.88 if num_metrics > 0 else 1.0 # Leave space for legend if plots exist
    fig.tight_layout(rect=[0, 0.03, right_margin, 0.93])
    save_figure(fig, output_filename)


# --- NEW: Function to plot L4 history comparisons ---
def plot_l4_comparison(l4_history_stats, output_filename_base, plot_styles):
    """Creates history plots comparing L4 scenarios."""
    print(f"\nGenerating L4 Comparison History Plots: {output_filename_base}...")
    if not l4_history_stats:
        print("  No L4 history data loaded, skipping plots.")
        return

    scenario_names = list(l4_history_stats.keys())
    num_scenarios = len(scenario_names)
    if num_scenarios == 0: return

    # Determine metrics available across scenarios (focus on standard ones first)
    available_metrics = set()
    max_gen = 0
    for stats in l4_history_stats.values():
        available_metrics.update(stats.keys())
        for metric_data in stats.values():
             if 'mean' in metric_data and metric_data['mean'] is not None:
                 max_gen = max(max_gen, len(metric_data['mean']))
    generations = np.arange(max_gen)

    plot_metrics = [m for m in HISTORY_METRICS_L4 if m in available_metrics and m != 'n_questions'] # Exclude n_questions for main plot
    num_plots = len(plot_metrics)

    if num_plots > 0:
        fig_main, axs_main = plt.subplots(1, num_plots, figsize=(6 * num_plots, 4.5), sharex=True, squeeze=False)
        axs_main = axs_main.flatten()

        for i, metric in enumerate(plot_metrics):
            ax = axs_main[i]
            info = METRICS_INFO.get(metric, {'label': metric.replace("_"," ").title()}) # Get refined label or generate one

            for scenario_name, stats in l4_history_stats.items():
                if metric in stats:
                    style = plot_styles.get(scenario_name, plot_styles['default'])
                    mean_hist = stats[metric].get('mean')
                    std_hist = stats[metric].get('std')

                    if mean_hist is not None:
                        # Ensure history length matches max_gen for plotting
                        current_gen = np.arange(len(mean_hist))
                        ax.plot(current_gen, mean_hist, label=scenario_name,
                                color=style.get('color'), linestyle=style.get('linestyle'),
                                linewidth=plt.rcParams['lines.linewidth']+0.5)
                        if std_hist is not None and std_hist.shape == mean_hist.shape:
                            std_hist_clean = np.nan_to_num(std_hist)
                            ax.fill_between(current_gen, mean_hist - std_hist_clean, mean_hist + std_hist_clean,
                                            color=style.get('color'), alpha=ERROR_ALPHA+0.05, linewidth=0)

            ax.set_xlabel("Generation", fontweight='bold')
            ax.set_ylabel(f"{info['label']}", fontweight='bold') # Removed "Mean" prefix
            ax.grid(True, linestyle=':', alpha=0.7)
            if i == num_plots - 1:
                ax.legend(title="Scenario", loc='center left', bbox_to_anchor=(1.05, 0.5), 
                         fontsize='medium', title_fontsize='medium')
            if 'Deceptive' in info['label']: ax.set_ylim(bottom=-0.02)
            if 'True Value' in info['label']: ax.axhline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.set_xlim(0, max_gen - 1 if max_gen > 0 else 1) # Set consistent xlim
            # Add bold title for each subplot
            ax.set_title(f"{info['label']}", fontweight='bold')

        fig_main.suptitle("Level 4 Scenario Comparison: Metric Histories", 
                        fontsize=plt.rcParams['figure.titlesize'], fontweight='bold')
        right_margin = 0.85 if num_plots > 0 else 1.0
        fig_main.tight_layout(rect=[0, 0.03, right_margin, 0.93])
        save_figure(fig_main, f"{output_filename_base}_histories")

    # Separate plot for n_questions history if available
    if 'n_questions' in available_metrics:
        fig_q, ax_q = plt.subplots(figsize=(7, 4.5))
        has_q_data = False
        max_q_gen = 0 # Track length specifically for n_questions plot
        for scenario_name, stats in l4_history_stats.items():
             if 'n_questions' in stats:
                 mean_hist = stats['n_questions'].get('mean')
                 if mean_hist is not None:
                     has_q_data = True
                     style = plot_styles.get(scenario_name, plot_styles['default'])
                     current_q_gen = np.arange(len(mean_hist))
                     max_q_gen = max(max_q_gen, len(mean_hist))
                     ax_q.plot(current_q_gen, mean_hist, label=scenario_name,
                             color=style.get('color'), linestyle=style.get('linestyle'),
                             linewidth=plt.rcParams['lines.linewidth']+0.5)
                     # Std dev for n_questions might be less informative, skip fill_between

        if has_q_data:
            ax_q.set_xlabel("Generation", fontweight='bold')
            ax_q.set_ylabel("Mean Number of Questions", fontweight='bold')
            ax_q.set_title("Level 4: Number of Questions Over Time", fontweight='bold')
            ax_q.grid(True, linestyle=':', alpha=0.7)
            ax_q.legend(title="Scenario", loc='best', fontsize='medium') # Adjust legend pos
            ax_q.set_xlim(0, max_q_gen -1 if max_q_gen > 0 else 1)
            fig_q.tight_layout()
            save_figure(fig_q, f"{output_filename_base}_n_questions_history")
        else:
            plt.close(fig_q) # Close if no data was plotted
# --- NEW: Functions to plot L4 Final State Comparisons ---

def plot_l4_final_bars(l4_final_stats, output_filename_base, plot_styles):
    """Creates grouped bar charts comparing final metrics across L4 scenarios."""
    print(f"\nGenerating L4 Final Metric Bar Charts: {output_filename_base}...")
    if not l4_final_stats:
        print("  No L4 final stats loaded, skipping plots.")
        return

    scenario_names = list(l4_final_stats.keys())
    num_scenarios = len(scenario_names)
    if num_scenarios == 0: return

    # Metrics to plot (excluding n_questions and cluster props for simple bars)
    metrics_to_plot = [m for m in METRICS_INFO if not METRICS_INFO[m].get('is_cluster', False)]
    num_metrics = len(metrics_to_plot)
    if num_metrics == 0: return

    fig, axs = plt.subplots(1, num_metrics, figsize=(5 + 3 * num_metrics, 5.5), sharey=False, squeeze=False)
    axs = axs.flatten()
    x_indices = np.arange(num_scenarios)

    for i, metric_key in enumerate(metrics_to_plot):
        ax = axs[i]
        info = METRICS_INFO[metric_key]
        means = []
        stds = []
        colors = []
        valid_scenario_names = [] # Keep track of scenarios with data for this metric

        for name in scenario_names:
            if name in l4_final_stats and metric_key in l4_final_stats[name]:
                mean_val = l4_final_stats[name][metric_key].get('mean', np.nan)
                std_val = l4_final_stats[name][metric_key].get('std', np.nan)
                # Check if mean/std are valid numbers before appending
                if not np.isnan(mean_val):
                    means.append(mean_val)
                    stds.append(np.nan_to_num(std_val)) # Replace NaN std with 0
                    colors.append(plot_styles.get(name, plot_styles['default']).get('color'))
                    valid_scenario_names.append(name)
                else:
                     # Optionally skip or plot NaN marker? For now, skip.
                     print(f"  Skipping scenario '{name}' for metric '{metric_key}' due to NaN mean.")
            else:
                print(f"  Metric '{metric_key}' or scenario '{name}' not found in final stats.")


        if not valid_scenario_names: # Skip plotting if no valid data for this metric
            ax.set_title(f"Final {info['label']} (No Data)", fontweight='bold')
            ax.text(0.5, 0.5, 'No Valid Data', ha='center', va='center', transform=ax.transAxes, 
                   color='red', fontsize=14, fontweight='bold')
            continue

        valid_x_indices = np.arange(len(valid_scenario_names))
        
        # Create bars with increased width and improved error bars
        bars = ax.bar(valid_x_indices, means, width=0.7, yerr=stds, color=colors,
               capsize=plt.rcParams['errorbar.capsize'] + 1, error_kw={'alpha': 0.8, 'lw': 1.5, 'capthick': 1.5})
        
        # Add value labels on top of bars
        for bar_idx, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02*max(means),
                   f'{height:.2f}', ha='center', va='bottom', fontweight='bold', 
                   fontsize=10, color=colors[bar_idx])

        ax.set_ylabel(f"Final {info['label']}", fontweight='bold', fontsize=12) # Removed redundant "Mean"
        ax.set_xticks(valid_x_indices)
        ax.set_xticklabels(valid_scenario_names, rotation=45, ha='right', fontweight='bold')
        ax.set_title(f"Final {info['label']}", fontweight='bold', fontsize=14)
        ax.grid(True, axis='y', linestyle=':', alpha=0.7)
        
        # Improve the background
        ax.set_facecolor('#f8f8f8')  # Light gray background
        
        if 'Deceptive' in info['label']: 
            ax.set_ylim(bottom=0)
        if 'True Value' in info['label']: 
            ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
            # Add light coloring to positive/negative regions
            y_min, y_max = ax.get_ylim()
            ax.axhspan(0, y_max, facecolor='lightgreen', alpha=0.1)
            ax.axhspan(y_min, 0, facecolor='mistyrose', alpha=0.1)

    fig.suptitle("Level 4 Scenario Comparison: Final Outcomes", fontsize=plt.rcParams['figure.titlesize'], fontweight='bold')
    fig.tight_layout(rect=[0, 0.05, 1, 0.94]) # Adjust layout
    save_figure(fig, f"{output_filename_base}_final_bars")


def plot_l4_final_scatter(l4_final_stats, output_filename_base, plot_styles):
    """Creates scatter plot of final Fitness vs Value across L4 scenarios."""
    print(f"\nGenerating L4 Final Fitness vs Value Scatter Plot: {output_filename_base}...")
    if not l4_final_stats:
        print("  No L4 final stats loaded, skipping plot.")
        return

    scenario_names = list(l4_final_stats.keys())
    num_scenarios = len(scenario_names)
    if num_scenarios == 0: return

    fig, ax = plt.subplots(figsize=(8, 7))
    fitness_info = METRICS_INFO.get('fitness', {})
    value_info = METRICS_INFO.get('value', {})

    plotted_something = False
    for name in scenario_names:
        if name in l4_final_stats and 'fitness' in l4_final_stats[name] and 'value' in l4_final_stats[name]:
            fit_mean = l4_final_stats[name]['fitness'].get('mean', np.nan)
            fit_std = l4_final_stats[name]['fitness'].get('std', np.nan)
            val_mean = l4_final_stats[name]['value'].get('mean', np.nan)
            val_std = l4_final_stats[name]['value'].get('std', np.nan)
            style = plot_styles.get(name, plot_styles['default'])

            if not np.isnan(fit_mean) and not np.isnan(val_mean):
                plotted_something = True
                ax.errorbar(fit_mean, val_mean, xerr=np.nan_to_num(fit_std), yerr=np.nan_to_num(val_std),
                            fmt=style.get('marker', 'o'), color=style.get('color'),
                            markersize=plt.rcParams['lines.markersize'] + 4, # Make markers larger
                            label=name, alpha=0.9, capsize=plt.rcParams['errorbar.capsize'], elinewidth=1.5)
                
                # Add text labels next to points
                ax.annotate(name, (fit_mean, val_mean), xytext=(10, 0), 
                           textcoords='offset points', fontweight='bold',
                           fontsize=10, color=style.get('color'))
            else:
                 print(f"  Skipping scenario '{name}' in scatter plot due to NaN mean fitness/value.")


    if plotted_something:
        ax.set_xlabel(f"Final {fitness_info.get('label', 'Fitness')}", fontweight='bold', fontsize=12) # Removed redundant "Mean"
        ax.set_ylabel(f"Final {value_info.get('label', 'True Value')}", fontweight='bold', fontsize=12) # Removed redundant "Mean"
        ax.set_title("Level 4 Scenarios: Final Fitness vs True Value", fontweight='bold', fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        # Use a bolder, cleaner look with thicker spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        # Add a subtle background color to distinguish quadrants
        ax.axhspan(0, ax.get_ylim()[1], facecolor='lightgreen', alpha=0.1)
        ax.axhspan(ax.get_ylim()[0], 0, facecolor='mistyrose', alpha=0.1)
        
        fig.tight_layout()
        save_figure(fig, f"{output_filename_base}_final_scatter")
    else:
        print("  No valid data points to plot for L4 final scatter.")
        plt.close(fig)
# --- Function to plot initial belief distribution (Needs to be defined) ---
def plot_initial_belief_dist(level_dir, config_params, filename_base, title_prefix):
    """Loads belief data from first NPZ in a dir and plots initial distribution."""
    print(f"\nGenerating Initial Belief Distribution plot for {title_prefix}...")
    npz_path = find_first_npz(level_dir) # Find any NPZ from this level's runs
    if not npz_path:
        print(f"  Could not find NPZ file in {level_dir} to plot initial beliefs.")
        return

    try:
        with np.load(npz_path, allow_pickle=True) as data:
            belief_values = data.get('belief_values')
            belief_alignments = data.get('belief_alignments')
            cluster_indices = data.get('cluster_indices') # Might be None for L1

            if belief_values is None or belief_alignments is None:
                print(f"  Error: belief_values or belief_alignments missing in {npz_path}")
                return

            cluster_params_list = config_params.get('BELIEF_GENERATION', {}).get('clusters')

            # Limit number of points plotted for clarity if it's huge
            num_points = len(belief_values)
            max_points_plot = 5000
            if num_points > max_points_plot:
                 sample_indices = np.random.choice(num_points, max_points_plot, replace=False)
                 belief_values = belief_values[sample_indices]
                 belief_alignments = belief_alignments[sample_indices]
                 if cluster_indices is not None:
                     # Ensure cluster_indices is indexable after sampling
                     cluster_indices = cluster_indices[sample_indices]

            fig, ax = create_belief_space_plot(
                 belief_values=belief_values,
                 belief_alignments=belief_alignments,
                 cluster_indices=cluster_indices,
                 cluster_params=cluster_params_list,
                 title=f'{title_prefix}Initial Belief Distribution',
                 alpha=0.4, # Lower alpha for dense plots
            )
            # Check if fig is None (can happen if create_belief_space_plot skips due to bad data)
            if fig is not None:
                fig.tight_layout()
                save_figure(fig, f"{filename_base}_initial_beliefs")
            else:
                 print(f"  Skipped saving initial belief plot due to plotting error.")


    except Exception as e:
        print(f"  Error plotting initial belief distribution from {npz_path}: {e}")


def plot_final_population_beliefs(run_dir, config_params, filename_base, title_prefix):
    """Loads data from the first NPZ in run_dir and plots the final population's belief distribution."""
    print(f"\nGenerating Final Population Belief plot for {title_prefix}...")
    npz_path = find_first_npz(run_dir) # Find run NPZ in the specific subdirectory
    if not npz_path:
        print(f"  Could not find NPZ file in {run_dir} to plot final population.")
        return

    try:
        with np.load(npz_path, allow_pickle=True) as data:
            belief_values = data.get('belief_values')
            belief_alignments = data.get('belief_alignments')
            cluster_indices_all = data.get('cluster_indices')
            final_population = data.get('final_population')

            if belief_values is None or belief_alignments is None or final_population is None:
                print(f"  Error: Required arrays missing in {npz_path}")
                return

            # Get unique belief indices present in the final population
            final_belief_indices = np.unique(final_population.flatten())
            # Ensure indices are valid
            valid_indices_mask = (final_belief_indices >= 0) & (final_belief_indices < len(belief_values))
            final_belief_indices = final_belief_indices[valid_indices_mask]

            if final_belief_indices.size == 0:
                 print(f"  Warning: No valid final belief indices found in {npz_path}")
                 return

            # Select the corresponding values, alignments, and cluster indices
            final_vals = belief_values[final_belief_indices]
            final_aligns = belief_alignments[final_belief_indices]
            final_cluster_indices = cluster_indices_all[final_belief_indices] if cluster_indices_all is not None else None

            cluster_params_list = config_params.get('BELIEF_GENERATION', {}).get('clusters')

            fig, ax = create_belief_space_plot(
                 belief_values=final_vals,
                 belief_alignments=final_aligns,
                 cluster_indices=final_cluster_indices,
                 cluster_params=cluster_params_list,
                 title=f'{title_prefix}Final Population Beliefs (Sample Run)',
                 alpha=0.5, # Slightly higher alpha for final pop
            )
            # Check if fig is None
            if fig is not None:
                fig.tight_layout()
                save_figure(fig, f"{filename_base}_final_pop_beliefs")
            else:
                 print(f"  Skipped saving final population belief plot due to plotting error.")


    except Exception as e:
        print(f"  Error plotting final population beliefs from {npz_path}: {e}")


# --- Main Execution ---

if __name__ == "__main__":
    print("*"*60)
    print(" Starting Comprehensive Analysis Figure Generation ".center(60, '*'))
    print("*"*60)

    os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

    # --- Level 1 Analysis (Basic Sweeps) ---
    print("\n" + "="*20 + " Level 1 Analysis " + "="*20)
    l1_dir = os.path.join(RESULTS_BASE_DIR, L1_DIR_NAME)
    l1_config_params = get_config_params(l1_dir)
    if os.path.isdir(l1_dir) and l1_config_params:
        # Plot Initial Belief Dist (if applicable)
        if l1_config_params.get('BELIEF_GENERATION',{}).get('type') != 'clustered':
            plot_initial_belief_dist(l1_dir, l1_config_params, "fig1_L1_initial_beliefs", "L1: ")

        # 1D Sweeps (rho, beta)
        for param_key in ["rho", "beta"]:
            sweep_info = ANALYSIS_FILES.get(param_key)
            if sweep_info:
                # Use L3 plot style default for consistency
                l1_sweep_data = load_sweep_data({'L1': l1_dir}, param_key)
                if 'L1' in l1_sweep_data:
                    plot_1d_sweep(l1_sweep_data['L1'], sweep_info['param_col'],
                                  METRICS_INFO, f"fig2_L1_{param_key}", title_prefix="L1: ")
            else:
                 print(f"Skipping Level 1 {param_key} sweep - analysis info not found.")

        # 2D Sweep (rho vs beta)
        l1_2d_data = load_2d_sweep_data(l1_dir)
        if l1_2d_data:
            # Ensure param names match expected for title generation if needed
            plot_2d_heatmaps(l1_2d_data, "fig3_L1_rho_beta_2D", title_prefix="L1: ")

    else:
        print(f"Warning: Level 1 directory or config not found at {l1_dir}. Skipping L1 plots.")

    # --- Level 2 Analysis (Clustered Landscape) ---
    print("\n" + "="*20 + " Level 2 Analysis " + "="*20)
    l2_dir = os.path.join(RESULTS_BASE_DIR, L2_DIR_NAME)
    l2_config_params = get_config_params(l2_dir)
    if os.path.isdir(l2_dir) and l2_config_params:
        # Plot Initial Belief Distribution (should be clustered)
        plot_initial_belief_dist(l2_dir, l2_config_params, "fig4_L2_initial_beliefs", "L2: ")

        # Get cluster names from this config
        cluster_names = get_cluster_names_from_params(l2_config_params)
        # Update METRICS_INFO with actual names if found
        if cluster_names:
            for i, name in enumerate(cluster_names):
                metric_key = f'cluster_prop_{i}'
                if metric_key in METRICS_INFO:
                    METRICS_INFO[metric_key]['label'] = f'{name} Prop.' # Use actual name

        # Sweeps relevant to L2 (e.g., beta, global_correlation, deceptive_prop)
        l2_swept_params = ["beta", "global_correlation", "deceptive_prop"] # Adjust if other sweeps were run for L2
        for param_key in l2_swept_params:
            sweep_info = ANALYSIS_FILES.get(param_key)
            if sweep_info:
                l2_sweep_data = load_sweep_data({'L2': l2_dir}, param_key)
                if 'L2' in l2_sweep_data:
                    # Standard metrics plot
                    plot_1d_sweep(l2_sweep_data['L2'], sweep_info['param_col'],
                                  METRICS_INFO, f"fig5_L2_{param_key}", title_prefix="L2: ")
                    # Cluster composition plot
                    if cluster_names:
                         plot_l2_cluster_composition(l2_sweep_data['L2'], sweep_info['param_col'],
                                                     cluster_names, f"fig6_L2_{param_key}")
            else:
                 print(f"Skipping Level 2 {param_key} sweep - analysis info not found.")

        # Optional: L2 2D sweep if run
        # l2_2d_data = load_2d_sweep_data(l2_dir)
        # if l2_2d_data: plot_2d_heatmaps(l2_2d_data, "fig_L2_2D", title_prefix="L2 ")

        # Plot Final Pop for representative condition (e.g., mid global_correlation)
        # Find a specific run directory, e.g., from the middle of a sweep
        example_run_dir_l2 = os.path.join(l2_dir, "global_correlation_0.0") # Example path, adjust as needed
        if os.path.isdir(example_run_dir_l2):
             plot_final_population_beliefs(example_run_dir_l2, l2_config_params,
                                           "fig7_L2_final_pop_example", "L2 (Mid Correlation): ")
        else:
             print(f"Could not find example run directory for L2 final pop plot: {example_run_dir_l2}")


    # --- Level 3 Analysis (Scenario Comparisons) ---
    print("\n" + "="*20 + " Level 3 Analysis " + "="*20)
    l3_dirs = {name: os.path.join(RESULTS_BASE_DIR, dirname)
               for name, dirname in L3_SCENARIO_DIR_NAMES.items()
               if os.path.isdir(os.path.join(RESULTS_BASE_DIR, dirname))} # Only include existing dirs

    if not l3_dirs:
         print("Warning: No Level 3 scenario directories found. Skipping L3 analysis.")
    else:
        print(f"Found L3 scenario directories: {list(l3_dirs.keys())}")
        # Assume sweeps were run within each scenario for beta & N_Questions
        beta_sweep_info = ANALYSIS_FILES.get("beta")
        nq_sweep_info = ANALYSIS_FILES.get("N_Questions")

        if beta_sweep_info:
            l3_beta_data = load_sweep_data(l3_dirs, "beta")
            if l3_beta_data:
                plot_sweep_comparison_lines(l3_beta_data, beta_sweep_info['param_col'],
                                            METRICS_INFO, "fig8_L3_compare_vs_beta", PLOT_STYLE_L3)
        else: print("Skipping Level 3 beta comparison - analysis info not found.")

        if nq_sweep_info:
            l3_nq_data = load_sweep_data(l3_dirs, "N_Questions")
            if l3_nq_data:
                plot_sweep_comparison_bars(l3_nq_data, nq_sweep_info['param_col'],
                                           METRICS_INFO, "fig9_L3_compare_vs_Nq", PLOT_STYLE_L3)
        else: print("Skipping Level 3 N_Questions comparison - analysis info not found.")




# NEW function to load final stats from individual NPZ files for L4 scenarios
def load_scenario_final_stats(scenario_dirs, metrics_to_get):
    """Loads the final value of specified metrics from NPZ files for L4 scenarios and computes stats."""
    scenario_final_stats = {}
    print("\n--- Loading final state data for L4 scenarios from NPZ files ---")

    for name, scenario_dir in scenario_dirs.items():
        if not os.path.isdir(scenario_dir):
            print(f"  Warning: Directory not found for scenario '{name}': {scenario_dir}")
            continue

        run_files = glob.glob(os.path.join(scenario_dir, "run_*_arrays.npz"))
        if not run_files:
            print(f"  Warning: No run NPZ files found for scenario '{name}' in {scenario_dir}")
            continue

        print(f"  Processing {len(run_files)} runs for final stats: {name}...")
        final_values_per_metric = {metric: [] for metric in metrics_to_get}

        for rf in run_files:
            try:
                with np.load(rf, allow_pickle=True) as data:
                    for metric in metrics_to_get:
                        hist_key = f'history_{metric}'
                        if hist_key in data:
                            hist_array = data[hist_key]
                            if hist_array.size > 0:
                                # Get the last element, handle potential multi-dim arrays (like clusters)
                                final_val = hist_array[-1]
                                final_values_per_metric[metric].append(final_val)
                            else:
                                final_values_per_metric[metric].append(np.nan) # Append NaN if history empty
                        else:
                            final_values_per_metric[metric].append(np.nan) # Append NaN if metric missing

            except Exception as e:
                print(f"    Error loading final value from {rf}: {e}")
                # Append NaNs for all metrics for this failed run
                for metric in metrics_to_get:
                    final_values_per_metric[metric].append(np.nan)

        # Compute stats (mean/std of final values) for this scenario
        stats = {}
        for metric, final_vals_list in final_values_per_metric.items():
             if final_vals_list: # Check if list is not empty
                 # Convert list to array for nanmean/nanstd
                 final_vals_array = np.array(final_vals_list)
                 with warnings.catch_warnings():
                      warnings.simplefilter("ignore", category=RuntimeWarning)
                      stats[metric] = {
                           'mean': np.nanmean(final_vals_array, axis=0), # axis=0 handles cluster arrays
                           'std': np.nanstd(final_vals_array, axis=0)
                      }
             else:
                 stats[metric] = {'mean': np.nan, 'std': np.nan} # Handle case where no runs had the metric

        scenario_final_stats[name] = stats

    return scenario_final_stats


# --- Level 4 Analysis (Dynamics and Test Design Scenarios) ---
print("\n" + "="*20 + " Level 4 Analysis " + "="*20)
# Define the specific path to the main Level 4 results directory
L4_BASE_RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, "level4_final_results") # Changed from L4_DIR_NAME to specific name

if not os.path.isdir(L4_BASE_RESULTS_DIR):
    print(f"Warning: Level 4 base directory not found: {L4_BASE_RESULTS_DIR}. Skipping L4 analysis.")
else:
    # Construct the full paths to the scenario subdirectories *within* L4_BASE_RESULTS_DIR
    l4_scenario_full_paths = {
        # Key: Scenario name (used for plotting labels/styles)
        # Value: Full path to the scenario's result folder
        name: os.path.join(L4_BASE_RESULTS_DIR, scenario_folder_name)
        for name, scenario_folder_name in L4_SCENARIO_DIR_NAMES.items()
    }

    # Filter this dictionary to include only scenarios whose folders actually exist
    l4_dirs_existing = {
        name: full_path
        for name, full_path in l4_scenario_full_paths.items()
        if os.path.isdir(full_path)
    }

    # Check if we found any valid L4 scenario directories
    if not l4_dirs_existing:
        print(f"Warning: No existing Level 4 scenario subdirectories found within {L4_BASE_RESULTS_DIR}.")
        print(f"Expected subfolders based on config: {list(L4_SCENARIO_DIR_NAMES.values())}")
        print("Skipping L4 analysis.")
   # --- Level 4 Analysis (Dynamics and Test Design Scenarios) ---
# ... (previous L4 code for loading directories) ...
    else:
        print(f"Found L4 scenario directories: {list(l4_dirs_existing.keys())}")
        # --- Load History Data ---
        l4_history_stats = load_scenario_history_data(l4_dirs_existing)

        if l4_history_stats:
            print("Successfully loaded history data for L4 scenarios.")
            # --- Plot Comparison Histories ---
            plot_l4_comparison(l4_history_stats, "fig10_L4_compare", PLOT_STYLE_L4)

            # --- Load Final State Data --- << NEW SECTION
            final_metrics_to_get = ['fitness', 'value', 'deceptive_ratio_raw']
            l4_final_stats = load_scenario_final_stats(l4_dirs_existing, final_metrics_to_get)

            if l4_final_stats:
                 print("Successfully loaded final state data for L4 scenarios.")
                 # --- Plot Final State Comparisons --- << NEW PLOTS
                 plot_l4_final_bars(l4_final_stats, "fig11_L4_compare", PLOT_STYLE_L4)
                 plot_l4_final_scatter(l4_final_stats, "fig12_L4_compare", PLOT_STYLE_L4)
            else:
                 print("Could not load final state data for L4 scenarios.")

            # --- Plot Final Population Beliefs (Example) --- << Keep this, maybe rename figure number
            # Choose a scenario to visualize the final belief state
            scenario_to_plot_name = "Combined_Dynamic_Mutation" # Or choose another key from l4_dirs_existing
            scenario_to_plot_dir = l4_dirs_existing.get(scenario_to_plot_name)

            if scenario_to_plot_dir:
                 # Config loading should check this specific scenario dir and its parent (L4_BASE_RESULTS_DIR)
                 l4_scenario_config = get_config_params(scenario_to_plot_dir)
                 if l4_scenario_config:
                     plot_final_population_beliefs(
                         scenario_to_plot_dir,
                         l4_scenario_config,
                         f"fig13_L4_final_pop_{scenario_to_plot_name}", # Renumbered figure
                         f"L4 ({scenario_to_plot_name}): "
                     )
                 else:
                     print(f"Could not load config params for {scenario_to_plot_dir} to plot final pop.")
            else:
                 print(f"Did not find '{scenario_to_plot_name}' scenario directory for final population plot.")
        else:
            print("Could not load sufficient history data to generate L4 comparison plots.")

# ... (rest of the main block) ..
# --- Final message --- # (This part was already outside the L4 block)
print("\n" + "*"*60)
print(" Analysis Figure Generation Script Finished ".center(60, '*'))
print(" Figures saved in: ".center(60, ' '))
print(f" {os.path.abspath(FIGURE_OUTPUT_DIR)} ".center(60, ' '))
print("*"*60)