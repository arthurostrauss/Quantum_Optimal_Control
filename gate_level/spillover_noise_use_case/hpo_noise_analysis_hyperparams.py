import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import product
from fractions import Fraction
from typing import Union, List, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.dirname(os.path.dirname(current_dir))
if module_path not in sys.path:
    sys.path.append(module_path)

# Assuming helper_functions.py is correctly placed and contains the necessary functions
from rl_qoc.helper_functions import load_from_pickle

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",  # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


# Collect filenames from a directory
def collect_filenames(
    directory: Union[List[str], str], date: Optional[str] = None
) -> List[str]:
    filenames = []
    if isinstance(directory, str):
        directory = [directory]
    for dir in directory:
        for root, _, files in os.walk(dir):
            for file in files:
                if file.endswith(".pickle") and (date is None or date in file):
                    filenames.append(os.path.join(root, file))
    return filenames


# Extract parameters from filename
def extract_phi_gamma(filename: str) -> tuple:
    parts = filename.split("/")
    params_str = parts[-1]
    params_parts = params_str.split("_")
    phi_val = next(part.split("-")[1] for part in params_parts if "phi" in part)
    gamma_val = next(part.split("-")[1] for part in params_parts if "gamma" in part)
    return phi_val, gamma_val


# Process a single file to extract data
def process_file(filename):
    phi, gamma = extract_phi_gamma(filename)
    data = load_from_pickle(filename)
    training_results = data["training_results"]
    return phi, gamma, training_results


# Create and animate a scatter plot
def create_animation(file_name: str):
    data_list = load_from_pickle(file_name)
    phi_val, gamma_val = extract_phi_gamma(file_name)

    colors = ["b", "g", "r", "c", "m", "y", "k"]
    markers = ["o", "d", "s", "h"]

    fig, ax = plt.subplots()
    # Create a scatter plot with empty data
    scat = ax.scatter([], [])
    counter_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    # Set axes limits
    ax.set_xlim(
        [
            1,
            1.2
            * max(
                [
                    data["hyper_params"]["MINIBATCH_SIZE"]
                    * data["hyper_params"]["BATCHSIZE_MULTIPLIER"]
                    for data in data_list
                ]
            ),
        ]
    )
    ax.set_ylim(
        [
            1,
            1.2
            * max(
                [
                    data["hyper_params"]["N_SHOTS"]
                    * data["hyper_params"]["SAMPLE_PAULIS"]
                    for data in data_list
                ]
            ),
        ]
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    phi_fraction = Fraction(phi_val[:-2]).limit_denominator()
    numerator, denominator = phi_fraction.numerator, phi_fraction.denominator
    if numerator != 1 and denominator != 1:
        ax.set_title(f"$\\phi$ = {numerator}π/{denominator}, gamma = {gamma_val}")
    elif denominator != 1:
        ax.set_title(f"$\\phi$ = π/{denominator}, gamma = {gamma_val}")
    else:
        ax.set_title(f"$\\phi$ = π, gamma = {gamma_val}")

    # Create an initialization function that clears the data
    def init():
        scat.set_offsets(np.empty((0, 2)))
        counter_text.set_text("")
        ax.plot(
            [
                0,
                max(
                    [
                        data["hyper_params"]["N_SHOTS"]
                        * data["hyper_params"]["SAMPLE_PAULIS"]
                        for data in data_list
                    ]
                ),
            ],
            [
                0,
                max(
                    [
                        data["hyper_params"]["N_SHOTS"]
                        * data["hyper_params"]["SAMPLE_PAULIS"]
                        for data in data_list
                    ]
                ),
            ],
        )
        ax.grid(True)
        return (scat,)

    # Create an update function that adds the data for the next dictionary
    def update(i: int):
        color = colors[i % len(colors)]
        marker = markers[i // len(colors) % len(markers)]

        data = data_list[i]
        x = (
            data["hyper_params"]["MINIBATCH_SIZE"]
            * data["hyper_params"]["BATCHSIZE_MULTIPLIER"]
        )
        y = data["hyper_params"]["N_SHOTS"] * data["hyper_params"]["SAMPLE_PAULIS"]
        ax.scatter(x, y, color=color, marker=marker, label=f"Trial {i + 1}")

        # counter_text.set_text('HPO trial ranking: #{}'.format(i+1))
        ax.legend(
            ncol=1,
            bbox_to_anchor=(1, 1),
            loc="upper left",
            fontsize="small",
            frameon=False,
        )
        plt.subplots_adjust(right=0.8)

        return (scat,)

    # Create the animation
    ani = FuncAnimation(
        fig,
        update,
        frames=range(len(data_list)),
        init_func=init,
        blit=True,
        interval=500,
    )

    return ani


##################### Hyperparameter Analysis #####################


def extract_parameters_hpo_analysis(
    target, fidelity_info, considered_fidelities, file_data
):
    considered = target in considered_fidelities
    achieved = considered and fidelity_info[target].get("achieved", False)
    shots_used = fidelity_info[target].get("shots_used", 0) if achieved else "N/A"
    updates_used = fidelity_info[target].get("update_at", 0) if achieved else "N/A"
    hardware_runtime = (
        fidelity_info[target].get("hardware_runtime", "N/A") if achieved else "N/A"
    )
    file_data[(f"fidelity: {target}", "considered")] = considered
    file_data[(f"fidelity: {target}", "achieved")] = achieved
    file_data[(f"fidelity: {target}", "shots_used")] = shots_used
    file_data[(f"fidelity: {target}", "updates_used")] = updates_used
    file_data[(f"fidelity: {target}", "shots_per_update")] = (
        int(np.ceil(shots_used / updates_used)) if achieved else "N/A"
    )
    file_data[(f"fidelity: {target}", "hardware_runtime_est [s]")] = (
        hardware_runtime if achieved else "N/A"
    )


def get_dataframe(directory_paths, specific_date):
    filenames = collect_filenames(directory_paths, specific_date)
    all_data = compile_data(filenames)
    df = prepare_dataframe(all_data)
    df.sort_index(inplace=True)
    return df


def process_file_hpo_analysis(filename):
    phi, gamma = extract_phi_gamma(filename)
    data = load_from_pickle(filename)[0]
    training_results = data["training_results"]
    fidelity_info = training_results["fidelity_info"]
    considered_fidelities = list(fidelity_info.keys())

    file_data = {
        ("batchsize", ""): data["hyper_params"].get("MINIBATCH_SIZE")
        * data["hyper_params"].get("BATCHSIZE_MULTIPLIER"),
        ("n_shots", ""): data["hyper_params"].get("N_SHOTS"),
        ("sample_paulis", ""): data["hyper_params"].get("SAMPLE_PAULIS"),
    }

    for target in [0.99, 0.999, 0.9999, 0.99999]:
        extract_parameters_hpo_analysis(
            target, fidelity_info, considered_fidelities, file_data
        )

    return (phi, gamma), file_data


def compile_data(filenames):
    all_data = {}
    for filename in filenames:
        try:
            index, file_data = process_file_hpo_analysis(filename)
            all_data[index] = file_data
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue
    return all_data


def prepare_dataframe(all_data):
    multi_index = pd.MultiIndex.from_tuples(next(iter(all_data.values())).keys())
    df = pd.DataFrame.from_dict(all_data, orient="index", columns=multi_index)
    df.index.names = ["phi", "gamma"]
    return df


def filter_data(df, phi, fidelity):
    """Filter DataFrame based on fidelity and phi."""
    subset_df = df.xs(key=(phi, slice(None)), level=(0, 1), drop_level=False)
    return subset_df[
        (subset_df[(f"fidelity: {fidelity}", "considered")] == True)
        & (subset_df[(f"fidelity: {fidelity}", "achieved")] == True)
    ]


def extract_data(subset_df, fidelity, scaling_factor):
    """
    Extracts and normalizes data from a subset DataFrame.

    Parameters:
        subset_df (pd.DataFrame): The subset of DataFrame from which to extract data.
        scaling_factor (float): The factor used to scale the shots_used data.

    Returns:
        tuple: A tuple containing batch sizes, normalized shots times sample paulis, normalized shots used, and gamma values.
    """
    batchsize = subset_df["batchsize"]
    n_shots_times_sample_paulis = subset_df["n_shots"] * subset_df["sample_paulis"]
    shots_used_normalized = (
        subset_df[(f"fidelity: {fidelity}", "shots_used")] / (scaling_factor)
    ).tolist()
    gamma_values = np.array(subset_df.index.get_level_values(1).astype(float))

    return batchsize, n_shots_times_sample_paulis, shots_used_normalized, gamma_values


def plot_threshold_line(ax, n_shots_times_sample_paulis):
    """
    Plots a threshold line between batchsize and n_shots_times_sample_paulis. (y = x)
    """
    ax.plot(
        [0, max(n_shots_times_sample_paulis)],
        [0, max(n_shots_times_sample_paulis)],
        color="red",
        linestyle="--",
    )


def format_scaling_factor(scaling_factor):
    """
    Formats a scaling factor into scientific notation for display purposes.

    Parameters:
        scaling_factor (float): The scaling factor to format.

    Returns:
        str: The formatted scaling factor in the form of x*10**y.
    """
    formatted_number = f"{scaling_factor:.1e}"  # Basic scientific notation

    # Split the formatted number on 'e' to separate the coefficient and exponent
    parts = formatted_number.split("e")
    coefficient = parts[0]
    exponent = int(parts[1])  # Convert exponent to an integer

    # Custom formatting to achieve the x*10**y format
    custom_formatted = f"{coefficient} \\times 10^{{{exponent}}}"
    return custom_formatted


def make_scatter(
    fidelity,
    ax,
    batchsize,
    n_shots_times_sample_paulis,
    shots_used_normalized,
    gamma_values,
    gamma_range,
    scaling_factor,
):
    """
    Creates a scatter plot on the provided axes and adds a custom legend with bubble sizes.

    Parameters:
        fidelity (str): The fidelity level being plotted.
        ax (matplotlib.axes.Axes): The axes on which to plot.
        batchsize (list): List of batch sizes.
        n_shots_times_sample_paulis (list): List of product of shots and sample paulis.
        shots_used_normalized (list): Normalized list of shots used.
        gamma_values (numpy.ndarray): Array of gamma values.
        gamma_range (list): Min and max range for gamma values.
        scaling_factor (float): Scaling factor used for normalizing shots used.

    Returns:
        matplotlib.collections.PathCollection: The scatter plot object.
    """
    marker = "o" if fidelity != "0.99" else "d"
    scatter = ax.scatter(
        x=batchsize,
        y=n_shots_times_sample_paulis,
        s=shots_used_normalized,
        marker=marker,
        c=gamma_values,
        cmap="viridis",
        alpha=0.7,
        vmin=gamma_range[0],
        vmax=gamma_range[1],
    )

    plot_threshold_line(ax, n_shots_times_sample_paulis)

    legend_sizes = np.quantile(shots_used_normalized, [0.25, 0.5, 0.75])
    labels = [f"{int(size)}" for size in legend_sizes]
    scatter_handles = [
        plt.scatter(
            [], [], s=size, color="gray", alpha=0.6, edgecolors="w", linewidth=1
        )
        for size in legend_sizes
    ]
    bubble_legend = ax.legend(
        scatter_handles,
        labels,
        title=f"total shots $[{format_scaling_factor(scaling_factor)}]$",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
        fontsize="small",
        ncol=len(labels),
    )  # legend(scatter_handles, labels, title='shots $[\\times 10^{5}]$', loc='upper left', frameon=True, fontsize='small')
    ax.add_artist(bubble_legend)  # Add bubble size legend back to each subplot

    return scatter


def make_title(ax, phi):
    """
    Sets the title of the axes based on the phi value.

    Parameters:
        ax (matplotlib.axes.Axes): The axes on which to set the title.
        phi (str): The phi value to format into the title.
    """
    phi_fraction = Fraction(phi[:-2]).limit_denominator()
    numerator, denominator = phi_fraction.numerator, phi_fraction.denominator
    if numerator != 1 and denominator != 1:
        ax.set_title(f"$\\phi$ = {numerator}π/{denominator}")
    elif denominator != 1:
        ax.set_title(f"$\\phi$ = π/{denominator}")
    else:
        ax.set_title(f"$\\phi$ = π")


def set_labels(ax, i):
    """
    Sets the labels and scales for the axes.
    """
    ax.set_xlabel("batchsize")
    ax.set_ylabel("n_shots * sample_paulis" if i == 0 else "")
    ax.set_xscale("log")
    ax.set_yscale("log")


def plot_hyperparam_analysis(df, fidelity_levels, phi_values, scaling_factor):
    """
    Plots the hyperparameter analysis for a given DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data for the analysis.
    - fidelity_levels (list): A list of fidelity levels to plot.
    - phi_values (list): A list of phi values to plot.
    - scaling_factor (float): The scaling factor for the analysis.

    Returns:
    - None

    This function creates a figure for each fidelity level and plots the hyperparameter analysis for the given DataFrame.
    It iterates over the fidelity levels and phi values to create subplots for each combination.
    The relevant data is extracted from the DataFrame and used to create scatter plots.
    The scatter plots show the relationship between various parameters and the spillover-rate gamma.
    A colorbar is added to the last subplot to represent the gamma values.

    Note: The function assumes that the necessary plotting libraries (e.g., matplotlib) have been imported.
    """

    # Determine the global range of gamma values for consistent color mapping
    gamma_values_global = df.index.get_level_values(1).unique()
    gamma_range = [gamma_values_global.min(), gamma_values_global.max()]

    # Create a figure for each fidelity level
    for fidelity in fidelity_levels:
        fig, axs = plt.subplots(1, 4, figsize=(20, 8), sharey=True)
        fig.suptitle(f"Fidelity Target: {fidelity}")

        for i, phi in enumerate(phi_values):
            ax = axs[i]
            ax.grid(True)

            # Filter DataFrame for the current fidelity and phi, where considered and achieved are True
            subset_df = filter_data(df, phi, fidelity)
            # Skip if empty
            if subset_df.empty:
                continue

            # Extract relevant data
            (
                batchsize,
                n_shots_times_sample_paulis,
                shots_used_normalized,
                gamma_values,
            ) = extract_data(subset_df, fidelity, scaling_factor)

            scatter = make_scatter(
                fidelity,
                ax,
                batchsize,
                n_shots_times_sample_paulis,
                shots_used_normalized,
                gamma_values,
                gamma_range,
                scaling_factor,
            )

            subset_df[("efficiency", "")] = 1 / pd.to_numeric(
                subset_df[(f"fidelity: {fidelity}", "shots_used")], errors="coerce"
            ).replace({0: np.nan})

            make_title(ax, phi)
            set_labels(ax, i)

        # Add a colorbar to the last subplot, linking it to the gamma values, but only once per fidelity level
        cbar = fig.colorbar(
            scatter,
            ax=axs.ravel().tolist(),
            orientation="horizontal",
            pad=0.25,
            aspect=40,
            shrink=0.75,
            label="spillover-rate $\gamma$",
        )

        plt.show()


def animate_hpo_outcomes_scatter(animation_file_path: str):
    animation = create_animation(animation_file_path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name_animation = os.path.join(script_dir, "hyperparam_animation.gif")
    animation.save(file_name_animation, writer="Pillow")

    logging.warning("Saved animation to {}".format(file_name_animation))


# Main function to run the workflow
def main(
    phis: np.array,
    gammas: np.array,
    animation_file_path: str,
    directory_paths: List[str],
    specific_date: Optional[str],
    fidelity_levels: List[float],
    phi_values: List[str],
    scaling_factor: float,
):
    animate_hpo_outcomes_scatter(animation_file_path)

    # Hyperparameter analysis
    df = get_dataframe(directory_paths, specific_date)
    plot_hyperparam_analysis(df, fidelity_levels, phi_values, scaling_factor)


if __name__ == "__main__":
    ### Infidelity analysis ###
    phis = np.pi * np.array([1])
    gammas = np.linspace(0.01, 0.15, 15)

    ### Animation of HPO outcomes ###
    animation_file_path = "/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/gate_level/spillover_noise_use_case/hpo_results/resource_constraint/max_hardware_runtime-600_07-05-2024/phi-0.5pi_gamma-0.01_maxruntime-600_custom-cost-value--15160.216284_timestamp_06-05-2024_22-28-05.pickle"

    ### Hyperparameter analysis ###
    directory_paths = [
        "/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/gate_level/spillover_noise_use_case/hpo_results/resource_constraint/max_hardware_runtime-600_07-05-2024"
    ]
    specific_date = None

    fidelity_levels = [0.999, 0.9999, 0.99999]
    phi_values = ["0.25pi", "0.5pi", "0.75pi", "1.0pi"]
    scaling_factor = 5e4

    main(
        phis,
        gammas,
        animation_file_path,
        directory_paths,
        specific_date,
        fidelity_levels,
        phi_values,
        scaling_factor,
    )
