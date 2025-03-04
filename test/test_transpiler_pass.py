from rl_qoc.helpers.transpiler_passes import CustomGateReplacementPass
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.transpiler import PassManager


def test_custom_gate_replacement_pass():
    # Define the custom gate
    qr = QuantumRegister(2, "qr")
    custom_gate = lambda qc, param, qarg: qc.rx(param[0], qarg)

    # Define the circuit
    circuit = QuantumCircuit(qr)
    circuit.ry(0.2, qr[0])

    # Define the pass
    pass_ = CustomGateReplacementPass(
        target_instructions=("ry", (0,)), new_elements=custom_gate, parameters=[[0.3]]
    )

    # Run the pass
    pm = PassManager(pass_)
    circuit = pm.run(circuit)
    print(circuit)


test_custom_gate_replacement_pass()
