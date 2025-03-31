
# Simulating the Evolution of AI Alignment and Values

This project simulates the evolutionary dynamics of AI models under selection pressure based on alignment tests. It explores the potential divergence between tested alignment signals (`a`) and the models' underlying true values (`v`), particularly the risk of "alignment mimicry" where models appear aligned without possessing genuinely beneficial values.

The simulation framework is inspired by the concepts outlined in ["Modelling the evolution of alignment and values in machine intelligence"](https://elsworth.phd/Formalisms/Analytical-Research/Evolution-of-Alignment-and-Values).

Important note: I forgot to update the base deceptiveness ratio from 0.4 to 0.3 in the sweeps for level 2 and 3. My poor computer is already crying from running all these simulations, so I'm not going to re-run them. I fixed the config files, but the existing data in the repo is still off. It shouldn't be a huge deal, but it is important to note.

## Overview

The simulation models a population of AI agents, each possessing a set of discrete "beliefs". Each belief has an intrinsic **true value** (`v`) representing its real-world impact (positive or negative) and an **alignment signal** (`a`) representing how it performs on an alignment test (`Q`).

Key dynamics explored:

* **Imperfect Correlation:** The alignment signal (`a`) and true value (`v`) are often correlated (e.g., using a bivariate normal distribution with correlation `rho`), but this correlation is imperfect ($\rho < 1$).
* **Selection:** Models are selected for reproduction based on their performance on the alignment test (`Q`), calculated from the alignment signals (`a`) of their beliefs activated by the test questions. Selection pressure is controlled by `beta`.
* **Evolution:** Selected models reproduce (with optional mutation), passing their beliefs to the next generation.
* **Divergence:** The simulation investigates conditions (e.g., low `rho`, high `beta`, specific belief landscapes, test properties) under which the population evolves high alignment fitness (`F`) but low or negative average true value (`U`), leading to deceptive alignment.

## Features & Simulation Levels

The simulation code is structured progressively through levels:

* **Level 0:** Minimal viable simulation demonstrating the core divergence effect with a simple bivariate normal belief distribution and basic inheritance.
* **Level 1:** Introduces parameter sweeps (e.g., over `rho`, `beta`), statistical analysis over multiple runs (`N_RUNS`), parallelization, and 2D parameter scans. Calculates the "Deceptive Belief Ratio".
* **Level 2:** Implements more complex belief landscapes using multiple clusters (n-modal distributions), allowing exploration of how landscape structure affects evolution (e.g., capture by a "Deceptive" cluster).
* **Level 3:** Adds enhanced realism through mechanisms like mutation during reproduction and more sophisticated activation models (e.g., similarity-based). Compares these scenarios.
* **Level 4:** Focuses on exploring the impact of test design and dynamics.
    * Models different static test properties (Coverage, Question Correlation) via different `ACTIVATION_TYPE` settings.
    * Simulates dynamic evaluation strategies:
        * `dynamic_test`: Periodically adds new questions targeting common deceptive beliefs.
        * `improving_alignment`: Simulates evaluators getting better at assigning alignment scores over time.
    * Compares these scenarios, typically defined in a YAML configuration file (`LEVEL4_SCENARIOS`).


## Requirements

* Python 3.8+
* NumPy
* Matplotlib
* PyYAML
* (Optional but recommended for faster saving: `pandas` - used implicitly by some older `np.savez` versions if saving complex objects, although current script seems fine without it)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or manually: pip install numpy matplotlib pyyaml
    ```

    *(Create a `requirements.txt` file with `numpy`, `matplotlib`, `pyyaml`)*

## Configuration

Simulations are configured using YAML files located in the `configs/` directory. Key parameters include:

* `LEVEL`: The simulation level to run (0-4). Determines which execution logic is used.
* `N_RUNS`: Number of independent simulation runs per parameter setting or scenario (crucial for statistical significance).
* `N_GENERATIONS`: Duration of each simulation run.
* `POPULATION_SIZE`: Number of models in the population.
* `N_BELIEFS_TOTAL`, `N_BELIEFS_PER_MODEL`: Size of the belief space and complexity of models.
* `BELIEF_GENERATION`: Defines the belief landscape (e.g., `type: "clustered"` with `clusters` list, or `type: "bivariate"` with `rho`). Includes parameters for dynamic alignment improvement (`target_rho`, `improvement_rate`).
* `SELECTION_PRESSURE_BETA`: Strength of selection based on alignment fitness.
* `ACTIVATION_TYPE`: Model for how tests activate beliefs (`simple`, `coverage`, `correlation`, `similarity`).
* `ACTIVATION_PARAMS`: Parameters specific to the chosen `ACTIVATION_TYPE` (e.g., `density`, `coverage`, `correlation`).
* `REPRODUCTION`, `MUTATION_RATE`: Method for generating the next generation (`inheritance` or `mutation`).
* `PARAMETER_SWEEPS` (Used by Level 1/2): Defines parameters to sweep over multiple values.
* `LEVEL4_SCENARIOS` (Used by Level 4): Defines different experimental setups (combinations of activation, dynamics, mutation) to compare.
* `DYNAMIC_FEATURES`: List of dynamic features to enable (e.g., `["dynamic_test", "improving_alignment"]`).
* `DYNAMIC_ACTIVATION_INTERVAL`, `ALIGNMENT_IMPROVEMENT_INTERVAL`: Frequency of dynamic updates.

See `configs/publication_run_level4.yaml` for a detailed example configured for robust Level 4 comparisons.

## Usage: Running Simulations

Execute the main simulation script from the command line, specifying the configuration file.

```bash
# Run Level 4 using the publication config (Level is set inside the YAML)
python simulation.py --config configs/publication_run_level4.yaml

# Explicitly specify level (overrides YAML) and use a different config
python simulation.py --config configs/level1_base.yaml --level 1

# Override specific parameters for a quick test run
python simulation.py --config configs/publication_run_level4.yaml --n_runs 5 --N_GENERATIONS 50 --output_dir quick_l4_test
```

* The `--config` argument specifies the YAML configuration file.
* The `--level` argument overrides the `LEVEL` setting in the config file.
* Other arguments like `--n_runs`, `--beta`, `--mutation_rate`, `--output_dir` can override specific config parameters.
* Results are saved by default to a timestamped directory within `./results/` (e.g., `results/level4_YYYYMMDD_HHMMSS`). Using `--output_dir <name>` saves results to `./results/<name>`.

## Usage: Generating Figures

After running the simulations, use the `create_paper_figures.py` script to generate plots from the saved results.

```bash
python create_paper_figures.py
```

* The script automatically looks for results directories (e.g., `level1_final_results`, `level2_final_results`, `level4_final_results`) inside the `./results` directory (configurable via `RESULTS_BASE_DIR` within the script).
* It loads aggregated data (CSVs from analysis subdirs) or raw run data (NPZs for L4 histories) as needed.
* Generated figures (PNG and optionally PDF) are saved to the `./paper_figures` directory (configurable via `FIGURE_OUTPUT_DIR` within the script).

## Output Description

* **`./results/<run_directory>/`**:
    * **`effective_config.yaml`**: The actual parameters used for the run (including overrides).
    * **Scenario Subdirectories (e.g., `scenario_Baseline/`, `scenario_Mutation/`) or Parameter Subdirectories (e.g., `rho_0.1/`)**: Contain individual run data.
        * **`run_<seed>_arrays.npz`**: NumPy archive containing array data (belief values/alignments, history arrays for fitness/value/etc., final population).
        * **`run_<seed>.json`**: Minimal JSON with basic run info (like initial deceptive ratio).
    * **Analysis Subdirectories (e.g., `analysis_rho/`, `level4_comparison_analysis/`)**: Contain aggregated results (like `rho_final_stats.csv`) or analysis plots generated *during* the run or analysis phase of the main script (distinct from `create_paper_figures.py`).
* **`./paper_figures/`**: Contains the final publication-quality figures generated by `create_paper_figures.py`. Filenames indicate the content (e.g., `fig2_L1_rho_vs_param.png`, `fig10_L4_compare_histories.png`).

## Interpreting Results

* **Fitness vs Value Divergence:** Look for scenarios where Mean Alignment Fitness increases while Mean True Value decreases or stays low (e.g., `fig2_L1_rho_vs_param.png`, `fig10_L4_compare_histories.png`).
* **Impact of `rho`:** Analyze how the correlation between alignment and value affects the final True Value and Deceptive Belief Ratio (e.g., `fig2_L1_rho_vs_param.png`, `fig3_L1_rho_beta_2D.png`). Low `rho` is expected to worsen outcomes.
* **Impact of `beta`:** Analyze how selection pressure affects convergence speed and final states (e.g., `fig2_L1_beta_vs_param.png`, `fig3_L1_rho_beta_2D.png`).
* **Deceptive Beliefs:** Track the Deceptive Belief Ratio. High final ratios indicate "alignment mimicry" (e.g., `fig2_L1_rho_vs_param.png`, `fig3_L1_rho_beta_2D.png`).
* **Belief Space:** Examine initial and final belief distributions (`fig1_...`, `fig4_...`, `fig7_...`, `fig13_...`) to see which types of beliefs (Benign, Neutral, Deceptive) dominate the population under different conditions.
* **Level 2 (Clusters):** Analyze cluster composition plots (`fig6_...`) to see if the population gets captured by specific belief clusters (especially the Deceptive one).
* **Level 3 (Scenarios):** Compare the effects of mutation and different activation models (`fig8_...`, `fig9_...`).
* **Level 4 (Dynamics):** Evaluate the effectiveness of dynamic testing and improving alignment by comparing final outcomes (e.g., `fig11_...`, `fig12_...`) and historical trends (`fig10_...`) against baseline and static test scenarios. Check if `dynamic_test` suppresses deceptive beliefs or if `improving_alignment` leads to higher true values.
