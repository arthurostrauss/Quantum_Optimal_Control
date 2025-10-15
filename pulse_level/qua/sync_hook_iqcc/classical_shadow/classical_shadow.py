import warnings
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qualang_tools.results import DataHandler

from iqcc_calibration_tools.quam_config.components import Quam, Transmon
from qm.qua import *
from qm import SimulationConfig, QuantumMachinesManager, generate_qua_script, Program
from qm.jobs.running_qm_job import RunningQmJob
from qm.jobs.simulated_job import SimulatedJob
from .shadow_config import ShadowConfig

from qualang_tools.units import unit
from qualang_tools.loops import qua_linspace
from ..xeb.macros import qua_declaration, reset_qubit, binary

u = unit(coerce_to_integer=True)

class ClassicalShadow:
    def __init__(
        self,
        config: ShadowConfig,
        machine: Quam,
    ):
        """
        Initialize the ClassicalShadow experiment.

        Args:
            config (ShadowConfig): Configuration for the classical shadow experiment.
            machine (QuAM): QuAM object containing the qubits and qubit pairs used in the experiment.
        """
        self.config = config
        self.machine = machine
        self.data_handler = DataHandler(name="classical_shadow",
                                        root_data_folder=self.config.save_dir)
        
    
    def cs_prog(self, simulate: bool = False) -> Program:
        
        """
        Generate the QUA script for the classical shadow experiment.
        """
        n_qubits = self.config.n_qubits
        dim = self.config.dim
        random_gates = len(self.config.measurement_basis)
        ge_thresholds = [qubit.resonator.operations[self.config.readout_pulse_name].threshold 
                         for qubit in self.config.qubits]
        
        with program() as cs_prog:
            I, I_st, Q, Q_st = qua_declaration(n_qubits=n_qubits, 
                                               readout_elements=[qubit.resonator for qubit in self.config.qubits],)
            random_basis = declare(int, size=n_qubits)
            random_basis_stream = declare_stream()
            state = declare(bool, size=self.config.n_qubits)
            state_int = declare(int, value=0)
            state_int_stream = declare_stream()
            r = Random(seed=self.config.seed)
            i = declare(int)
            j = declare(int)
            shot = declare(int)
            angle = declare(fixed)
            if self.config.gate_indices is not None:
                gate_indices = [declare(int,
                                         value=self.config.gate_indices[:, n].tolist()) for n in range(n_qubits)]
            self.machine.apply_all_flux_to_min()
            self.machine.apply_all_couplers_to_min()
            
            if simulate:
                for qubit in self.config.qubits:
                    qubit.xy.update_frequency(0)
            with for_(*qua_linspace(angle, 0., np.pi, self.config.num_angles)):
                with for_(i, 0, i < self.config.shadow_size, i + 1):
                    # Possible wait time before the experiment
                    # wait(...)
                    if self.config.gate_indices is not None:
                        for n in range(n_qubits):
                            assign(random_basis[n], gate_indices[n][i])
                            save(random_basis[n], random_basis_stream)
                    else:
                        # Sample random basis (assumed to be local measurements)
                        with for_(j, 0, j < n_qubits, j + 1):
                            assign(random_basis[j], r.rand_int(random_gates))
                            save(random_basis[j], random_basis_stream)

                    with for_(shot, 0, shot < self.config.shots_per_snapshot, shot + 1):
                        # Prepare state
                        if self.config.input_state_prep_macro_kwargs: #is not None:
                            self.config.input_state_prep_macro(angle, **self.config.input_state_prep_macro_kwargs)
                        else:
                            self.config.input_state_prep_macro()
                        align()
                        # for q, qubit, in enumerate(self.config.qubits):
                        #     with switch_(random_basis[q], unsafe=False):
                        #         # Apply the random basis rotation
                        #         for k in range(random_gates):
                        #             with case_(k):
                        #                 self.config.measurement_basis[k].gate_macro(qubit)
                        # align()
                        # Readout
                        for q, qubit, in enumerate(self.config.qubits):
                            # Replace switch case with conditional plays of the measurement basis rotations
                            qubit.xy.play("x90", condition=random_basis[q] == 0)
                            qubit.xy.play("-y90", condition=random_basis[q] == 1)
                            
                            
                        # Play the readout on the other resonator to measure in the same condition as when optimizing readout
                        for other_qubit in self.config.readout_qubits:
                            if other_qubit.resonator not in [qubit.resonator for qubit in self.config.qubits]:
                                other_qubit.resonator.play("readout")
                        for q, qubit, in enumerate(self.config.qubits):
                            # qubit.align()
                            qubit.resonator.measure(self.config.readout_pulse_name,
                                                    qua_vars=(I[q], Q[q]))
                            # State Estimation: returned as integer
                            assign(state[q], I[q] > ge_thresholds[q])
                            assign(state_int, state_int + (1<<q) * Cast.to_int(state[q]))

                            reset_qubit(self.config.reset_method,
                                        qubit,
                                        threshold=ge_thresholds[q],
                                        **self.config.reset_kwargs)
                        save(state_int, state_int_stream)
                        assign(state_int, 0)
                save(angle, "angle")
                
            with stream_processing():
                random_basis_stream.buffer(self.config.shadow_size, n_qubits).save_all("random_basis")
                state_int_stream.buffer(self.config.shadow_size, self.config.shots_per_snapshot).save_all("state_int")
        
        return cs_prog
    
    def run(self, simulate: bool = False,
            simulation_config: Optional[SimulationConfig] = None,
            qmm_cloud_simulator: Optional[QuantumMachinesManager] = None,
            **simulate_kwargs):
        config = self.machine.generate_config()
        if simulation_config is None:
            simulation_config = SimulationConfig(
                duration=10_000
            )
        cs_prog = self.cs_prog(simulate=simulate)
        if simulate and qmm_cloud_simulator is not None:
            qmm = qmm_cloud_simulator
        else:
            qmm = self.machine.connect()
        
        qm = qmm.open_qm(config)
        if simulate:
            with open("debug.py", "w+") as f:
                f.write(generate_qua_script(cs_prog, config))
            job = qm.simulate(cs_prog, simulate=simulation_config, **simulate_kwargs)
            
        elif self.config.generate_new_data:
            job = qm.execute(cs_prog)
        else:
            warnings.warn("No new data will be generated. Please set generate_new_data to True to generate new data.")
            return 
        
        return ClassicalShadowJob(job, self.config, self.data_handler)
        
            
class ClassicalShadowJob:
    def __init__(self, job: RunningQmJob | SimulatedJob, config: ShadowConfig, data_handler: DataHandler):
        """
        Initialize the ClassicalShadowJob object.

        Args:
            job (RunningQmJob | SimulatedJob): The job object returned by the QUA program.
            config (ShadowConfig): Configuration for the classical shadow experiment.
            data_handler (DataHandler): Data handler for saving and processing results.
        """
        self.job = job
        self._result_handles = self.job.result_handles
        self._result_handles.wait_for_all_values()
        self.config = config
        self.data_handler = data_handler
        self._gate_indices = np.zeros((self.config.num_angles, self.config.shadow_size, self.config.n_qubits), dtype=int)
        
        
    def _get_circuits(self):
        """
        Get the circuits from the job.
        """
        shadow_size = self.config.shadow_size
        gates = self._result_handles["random_basis"].fetch_all()['value']
        angles = np.linspace(0, np.pi, self.config.num_angles)
        self.config.input_state_prep_macro_kwargs["angle"] = angles
        input_state_circuit = self.config.input_state_circuit
        all_circuits = []
        for a, angle in enumerate(angles):
            for i in range(shadow_size):
                for j in range(self.config.n_qubits):
                    self._gate_indices[a, i, j] = gates[a][i][j]
                    
            circuits = [QuantumCircuit(self.config.n_qubits) for _ in range(shadow_size)]
            for i in range(shadow_size):
                circuits[i].compose(input_state_circuit(angle), inplace=True)
                for j in range(self.config.n_qubits):
                    circuits[i].append(self.config.measurement_basis[self._gate_indices[i, j]].gate,
                                    [j])
            all_circuits.append(circuits)
            return all_circuits
    
    def result(self):
        """
        Get the result of the job.
        """
        state_ints = self._result_handles["state_int"].fetch_all()['value']
        angles = self._result_handles["angle"].fetch_all("value")
        bitstrings = []
        for a in range(len(angles)):
            bitstrings.append([])
            for i, state_int in enumerate(state_ints):
                # Count all occurences of each bitstring and build a dictionary of counted bitstrings
                counts = {binary(i, self.config.n_qubits): 0 for i in range(self.config.dim)}
                for j in range(len(state_int)):
                    bitstring = binary(state_int[j], self.config.n_qubits)
                    counts[bitstring] += 1
                bitstrings[a].append(counts)

        return [(angle, [(bitstring, self._gate_indices[i]) for i, bitstring in enumerate(bitstrings_)]) for angle, bitstrings_ in zip(angles, bitstrings)]
    
    
    def ideal_result(self):
        """
        Get the ideal results of the job.
        """
        circuits = self._get_circuits()
        results = []
        for a, circuit_list in enumerate(circuits):
            results.append([])
            for i, circuit in enumerate(circuit_list):
                state = Statevector(circuit)
                probs = state.probabilities_dict()
                results[a].append((probs, self._gate_indices[i]))
            
        return results
            
            
            
            
     