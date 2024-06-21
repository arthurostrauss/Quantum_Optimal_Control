from typing import Optional
import os
import sys
import numpy as np
import time
from itertools import product

import cma
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.circuit.library import RXGate, IGate
import qiskit_aer.noise as noise
from qiskit_aer import AerSimulator
from qiskit.quantum_info.operators.measures import average_gate_fidelity, state_fidelity
from qiskit_experiments.library.tomography.basis import PauliPreparationBasis

current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.dirname(os.path.dirname(current_dir))
if module_path not in sys.path:
    sys.path.append(module_path)

from rl_qoc.helper_functions import save_to_pickle
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",  # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


def get_summary(
    optimized_fidelity,
    optimal_noise_free_params,
    optimized_params,
    optimal_deviations,
    end_time,
):
    return {
        "optimized_fidelity": optimized_fidelity,
        "optimal_noise_free_params": optimal_noise_free_params,
        "optimized_params": optimized_params,
        "optimal_deviations": optimal_deviations,
        "time_taken": end_time,
    }


def fidelity_function(params, phi_val, backend):
    """
    Calculate the negative fidelity between the ideal and noisy output states.

    Parameters:
        params (list): List of parameters for the U gates and RZX gate.
        phi_val (float): The phi value for the RX gate.
        gamma_val (float): The gamma value for the RX gate when model is 'noisy'.
        ideal_output_state (Statevector): The ideal output state.
        input_state (Statevector): The input state.

    Returns:
        float: The negative fidelity between the ideal and noisy output states.
    """

    input_circuits = [
        PauliPreparationBasis().circuit(s) for s in product(range(4), repeat=2)
    ]

    avg_gate_fid = []
    for input_circ in input_circuits:
        ideal_circ = get_ideal_circ(phi_val).compose(input_circ, front=True)
        ideal_circ = transpile(ideal_circ, backend, optimization_level=0)
        noisy_circ = get_noisy_circ(phi_val, params).compose(input_circ, front=True)
        noisy_circ = transpile(noisy_circ, backend, optimization_level=0)

        process_results = backend.run(noisy_circ).result()
        q_process_list = [
            process_results.data(0)["superop"],
            # process_results.data(0)['density_matrix'],
        ]
        avg_fidelity = np.mean(
            [
                average_gate_fidelity(q_process, Operator(ideal_circ))
                for q_process in q_process_list
                # state_fidelity(q_process, Statevector(ideal_circ))
                # for q_process in q_process_list
            ]
        )
        avg_gate_fid.append(avg_fidelity)

    return -np.mean(avg_fidelity)


def get_noisy_backend(gamma, phi):
    ident = Operator(IGate())
    rx_op = Operator(RXGate(gamma * phi))
    ident_rx_op = Operator(rx_op.tensor(ident))

    custom_rx_gate_label = "custom_kron(rx,ident)_gate"
    noise_model = noise.NoiseModel()

    coherent_crx_noise = noise.coherent_unitary_error(ident_rx_op)
    noise_model.add_quantum_error(coherent_crx_noise, [custom_rx_gate_label], [0, 1])
    noise_model.add_basis_gates(["unitary"])

    backend = AerSimulator(noise_model=noise_model)
    return backend


def get_cma_es_result(optimal_noise_free_params, phi_val, gamma_val, backend):
    """
    Use CMA-ES to optimize the fidelity function.
    """
    initial_sigma = 0.1
    es = cma.CMAEvolutionStrategy(
        optimal_noise_free_params,
        initial_sigma,
        {
            "maxiter": 1000,
            "tolx": 1e-6,
            "bounds": [-np.pi, np.pi],
        },
    )

    def wrapped_fidelity(params):
        return fidelity_function(params, phi_val, backend)

    start_time = time.time()
    result = es.optimize(wrapped_fidelity)
    end_time = time.time() - start_time
    logging.warning("Time taken for CMA-ES: {}s".format(round(end_time, 4)))

    optimized_params = result.best.x
    optimized_fidelity = (
        -result.best.f
    )  # Negate because fidelity_function returns negative fidelity
    optimal_deviations = optimal_noise_free_params - optimized_params

    return get_summary(
        optimized_fidelity,
        optimal_noise_free_params,
        optimized_params,
        optimal_deviations,
        end_time,
    )


def get_nelder_mead_result(optimal_noise_free_params, phi_val, backend):
    """
    Runs the Nelder-Mead optimization method to find the optimal parameters for a given fidelity function.

    Args:
        optimal_noise_free_params (array-like): The initial guess for the optimal noise-free parameters.
        phi_val (float): The phi value.
        backend: The backend used for the optimization.

    Returns:
        dict: A dictionary containing the summary of the optimization results, including the optimized fidelity,
              the optimal noise-free parameters, the optimized parameters, the optimal deviations, and the
              time taken for the optimization.
    """
    start_time = time.time()
    result = minimize(
        fidelity_function,
        optimal_noise_free_params,  # Initial guess
        args=(phi_val, backend),
        method="Nelder-Mead",  # Suitable for simple bounds and parameter tuning
        options={"maxiter": 1000, "xatol": 1e-6, "disp": False},
    )
    end_time = time.time() - start_time
    logging.warning("Time taken for Nelder-Mead: {}s".format(round(end_time, 4)))

    # Check the results
    optimized_params = result.x
    optimal_deviations = optimal_noise_free_params - optimized_params
    optimized_fidelity = -result.fun  # Negate to get the actual fidelity value

    return get_summary(
        optimized_fidelity,
        optimal_noise_free_params,
        optimized_params,
        optimal_deviations,
        end_time,
    )


def get_optimized_params(
    optimal_noise_free_params, phi_val, gamma_val, backend
) -> dict:
    """
    Optimize quantum circuit parameters for a given phi and gamma value using both Nelder-Mead and CMA-ES.

    Parameters:
        optimal_params (numpy.ndarray): Array of optimal parameters.
        phi_val (float): The phi value for the RX gate.
        gamma_val (float): The gamma value for the RX gate when model is 'noisy'.
        backend (AerSimulator): Backend for executing quantum circuits.

    Returns:
        dict: Dictionary containing optimization results.
    """
    results_NM = get_nelder_mead_result(optimal_noise_free_params, phi_val, backend)
    result_CMA = get_cma_es_result(
        optimal_noise_free_params, phi_val, gamma_val, backend
    )

    return {
        # Plugging in the optimal parameters for the noise-free case while having noise leads to the uncorrected fidelity
        "uncorrected_gate_fidelity": -fidelity_function(
            optimal_noise_free_params, phi_val, backend
        ),
        "nelder_mead": results_NM,
        "cma_es": result_CMA,
    }


def main(
    optimal_noise_free_params, phis, gammas, save_file_name: Optional[str] = None
) -> dict:
    """
    Main function to optimize the quantum circuit parameters to maximize state fidelity.

    Parameters:
        optimal_params (numpy.ndarray): Array of optimal parameters.
        phis (numpy.ndarray): Array of phi values.
        gammas (numpy.ndarray): Array of gamma values.
        save_file_name (str, optional): File name for saving results. Defaults to None.
        backend (AerSimulator): Backend for executing quantum circuits.

    Returns:
        dict: Dictionary containing optimization results.
    """

    result = {}
    for phi_val in phis:
        for gamma_val in gammas:
            gamma_val = round(gamma_val, 4)
            key = (phi_val, gamma_val)
            logging.warning(
                "Parameter Pair: phi = {}pi; gamma = {}".format(
                    phi_val / np.pi, gamma_val
                )
            )

            backend = get_noisy_backend(gamma_val, phi_val)
            result[key] = get_optimized_params(
                optimal_noise_free_params, phi_val, gamma_val, backend
            )

    if save_file_name is not None:
        # get the current directory of the script
        current_dir = os.path.dirname(os.path.realpath(__file__))
        save_file_path = os.path.join(current_dir, save_file_name)
        save_to_pickle(result, save_file_path)

    return result


def get_ideal_circ(phi: float):
    ideal_circ = QuantumCircuit(2, name=f"ideal_circ_{phi}pi")
    ideal_circ.rx(phi, 0)
    ideal_circ.cx(0, 1)
    return ideal_circ


def get_noisy_circ(phi, params):
    custom_rx_gate_label = "custom_kron(rx,ident)_gate"
    noisy_circ = QuantumCircuit(2)

    identity_op = Operator(IGate())
    rx_op = Operator(RXGate(phi))
    rx_2q_gate = Operator(identity_op.tensor(rx_op))
    noisy_circ.unitary(rx_2q_gate, [0, 1], label=custom_rx_gate_label)

    # Model custom CX gate
    noisy_circ.u(params[0], params[1], params[2], 0)
    noisy_circ.u(params[3], params[4], params[5], 1)
    noisy_circ.rzx(params[6], 0, 1)

    noisy_circ.save_superop()
    # circuit.save_density_matrix()
    return noisy_circ


if __name__ == "__main__":
    optimal_noise_free_params = np.pi * np.array([0.0, 0.0, 0.5, 0.5, -0.5, 0.5, -0.5])

    phis = np.pi * np.linspace(1 / 4, 1.0, 4)
    gammas = np.linspace(0.01, 0.15, 15)  # np.logspace(-4, -2, 3)

    save_file_name = "optimization_results_NM_CMAES.pickle"

    result = main(optimal_noise_free_params, phis, gammas, save_file_name)
    print(result)
