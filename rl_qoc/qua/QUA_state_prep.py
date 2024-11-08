"""
QUA program to be executed for arbitrary state prepation using model-free RL

Author: Arthur Strauss
Created on 19/12/2022
"""

# QUA imports
from QUA_config_two_sc_qubits import IBMconfig
from qualang_tools.bakery.bakery import baking
from qm.qua import *
import time
from qm.QuantumMachinesManager import QuantumMachinesManager


def qua_prog(n_qubits, param_table, batchsize, n_shots, schedule):
    with program() as model_free_rl:
        """Declaration of all necessary variables"""
        rep = declare(int)  # Iterator for sampling
        a = declare(int)
        timing = declare(int, value=0)

        I = declare(fixed, size=self._n_qubits)
        Q = declare(fixed, size=self._n_qubits)
        state_estimate = declare(bool, size=self._n_qubits)

        state_streams, I_streams, Q_streams = (
            [declare_stream()] * self._n_qubits,
            [declare_stream()] * self._n_qubits,
            [declare_stream()] * self._n_qubits,
        )
        state_strings, I_strings, Q_strings, q_strings, RR_strings = (
            ["state" + str(i) for i in range(self._n_qubits)],
            ["I" + str(i) for i in range(self._n_qubits)],
            ["Q" + str(i) for i in range(self._n_qubits)],
            ["q" + str(i) for i in range(self._n_qubits)],
            ["RR" + str(i) for i in range(self._n_qubits)],
        )

        mean_params = declare(fixed, size=n_actions)
        sigma_params = declare(fixed, size=n_actions)

        """Beginning of the experiment, infinite_loop is used. Each iteration consists in a new call in the
        Python of the evaluation of the cost function"""
        with infinite_loop_():
            pause()

            # Retrieve classical parameters for VQO

            with for_(a, init=0, cond=a < n_actions, update=a + 1):
                pause()
                assign(mean_params[a], IO1)
                assign(sigma_params[a], IO2)

            # Generate random selection of input states & tomography operations
            # for direct fidelity estimation

            with for_(
                rep, init=0, cond=rep <= N_shots, update=rep + 1
            ):  # Do N_shots times the same quantum circuit
                # Run the computation for each input state and tomography operator the parametrized process
                for state in states.keys():
                    for op in readout_ops.keys():
                        # Prepare input state
                        prepare_state(state)

                        # Apply parametrized U
                        apply_parametrized_pulse_sequence()  # QUA macro, using drawn actions to tune pulses in real
                        # time (e.g. amplitude, phase, frequency, truncations, whatever) could also be a simple b.run()

                        # Measurement and state determination

                        # Apply change of basis to retrieve Pauli expectation
                        change_basis(op)
                        for k in range(n):  # for each RR
                            measurement(RR_strings[k], I, Q)
                            state_saving(I, Q, state_estimate, state_streams[k])
                            # Active reset, flip the qubit if measured state is 1
                            with if_(state_estimate):
                                Rx(π, q_strings[k])
                            # Save I & Q variables, if deemed necessary by the user
                            raw_saving(I, Q, I_streams[k], Q_streams[k])

                    assign(timing, timing + 1)
                    save(timing, "timing")

        with stream_processing():
            for i in range(self._n_qubits):
                state_streams[i].boolean_to_int().buffer(
                    len(states.keys()), len(readout_ops.keys())
                ).save_all(state_strings[i])
                I_streams[i].buffer(
                    len(states.keys()), len(readout_ops.keys())
                ).save_all(I_strings[i])
                Q_streams[i].buffer(
                    len(states.keys()), len(readout_ops.keys())
                ).save_all(Q_strings[i])

    return model_free_rl_state_prep


job = QM.execute(VQGO)


def encode_params_in_IO(rot_angles, s_params):
    # Insert angles values using IOs by keeping track
    # of where we are in the QUA program
    for angle in rot_angles:
        while not (job.is_paused()):
            time.sleep(0.0001)
        QM.set_io1_value(angle)
        job.resume()
    for p in s_params:
        while not (job.is_paused()):
            time.sleep(0.0001)
        QM.set_io2_value(p)
        job.resume()


def AGI(params):  # Calculate cost function
    job.resume()
    rot_angles = params[0 : 3 * n * (d + 1)]
    s_params = params[3 * n * (d + 1) :]

    encode_params_in_IO(rot_angles, s_params)

    while not (job.is_paused()):
        time.sleep(0.0001)
    results = job.result_handles
    # time.sleep()
    output_states = []
    t = results.timing.fetch_all()["value"]
    print(t[-1])
    for m in range(n):
        output_states.append(results.get("state" + str(m)).fetch_all()["value"])

    counts = (
        {}
    )  # Dictionary containing statistics of measurement of each bitstring obtained
    expectation_values = {}

    for i in range(N_shots):
        bitstring = ""
        for s, st in list(enumerate(states.keys())):
            counts[st] = {}
            expectation_values[st] = {}
            for r, ope in list(enumerate(readout_ops.keys())):
                counts[st][ope] = {}
                expectation_values[st][ope] = {}
                for l in range(len(output_states)):
                    bitstring += str(output_states[l][i][s][r])
                if not (bitstring in counts[st][ope]):
                    counts[st][ope][bitstring] = 0
                    expectation_values[st][ope][bitstring] = 0
                else:
                    counts[st][ope][bitstring] += 1
                    if tomography_set[ope]["Ref"] == "ID__σ_z":
                        if bitstring == "00" or bitstring == "10":
                            expectation_values[st][ope][bitstring] += 1 / N_shots
                        elif bitstring == "01" or bitstring == "11":
                            expectation_values[st][ope][bitstring] -= 1 / N_shots
                    if tomography_set[ope]["Ref"] == "σ_z__σ_z":
                        if bitstring == "00" or bitstring == "11":
                            expectation_values[st][ope][bitstring] += 1 / N_shots
                        elif bitstring == "10" or bitstring == "01":
                            expectation_values[st][ope][bitstring] -= 1 / N_shots
                    if tomography_set[ope]["Ref"] == "σ_z__ID":
                        if bitstring == "00" or bitstring == "01":
                            expectation_values[st][ope][bitstring] += 1 / N_shots
                        elif bitstring == "10" or bitstring == "11":
                            expectation_values[st][ope][bitstring] -= 1 / N_shots

    # Here shall be implemented the calculation of the Average Gate Infidelity based on experimental results retrieved
    # above. It would amount to building the PTM representation of the process described and perform the calculation
    # indicated in original paper
    cost = 0
    return cost


init_rotation_angles = np.random.uniform(0, 2 * π, 3 * n * (d + 1))
init_source_params = np.random.uniform(0.0, 2.0, d)
init_angles = list(init_rotation_angles)
for param in init_source_params:
    init_angles.append(param)

Result = minimize(AGI, np.array(init_angles), method=optimizer)
print(Result)

job.halt()
