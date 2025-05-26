from iqcc_cloud_client import IQCC_Cloud
from rl_qoc.qua import QMEnvironment
from quam_libs.quam_builder.machine import QuAM
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter, Gate
from qiskit_qm_provider import FluxTunableTransmonBackend
import qiskit.pulse as qp
from typing import List

iqcc = IQCC_Cloud()

machine = QuAM.load()
backend = FluxTunableTransmonBackend(machine)
target_qubit = machine.active_qubits[0]
drive_channel = backend.get_pulse_channel(target_qubit.xy)
baseline_x_op = target_qubit.xy.operations["x180"].get_raw_value()


def apply_gate(qc: QuantumCircuit, params: List[Parameter], q_reg: QuantumRegister, **kwargs):

    custom_x = Gate("x_cal", 1, [params[0]], label="custom_x")
    qc.append(custom_x, [q_reg[0]])

    with qp.build() as pulse_program:
        qp.play(qp.Drag())
