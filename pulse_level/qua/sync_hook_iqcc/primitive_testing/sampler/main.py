from qiskit.circuit import QuantumCircuit, Parameter
from iqcc_cloud_client import IQCC_Cloud
from qiskit_qm_provider import FluxTunableTransmonBackend, InputType, QMSamplerV2, QMSamplerOptions
from qiskit import transpile
from qiskit_qm_provider.backend.backend_utils import add_basic_macros_to_machine
import numpy as np
from pathlib import Path
import json
from rl_qoc.qua.iqcc import get_machine_from_iqcc

# Set your quantum computer backend
path = Path.home() / "iqcc_token.json"
with open(path, "r") as f:
    iqcc_config = json.load(f)

backend_name = "gilboa"

machine, iqcc = get_machine_from_iqcc(backend_name, iqcc_config[backend_name])

add_basic_macros_to_machine(machine)
backend = FluxTunableTransmonBackend(machine)

qc = QuantumCircuit(1)
param = Parameter("param")
qc.rx(param, 0)
qc.measure_all()

param_values = np.linspace(0, np.pi, 10)
transpiled_circuits = transpile(qc, backend)

sampler = QMSamplerV2(backend, options=QMSamplerOptions(input_type=InputType.INPUT_STREAM))
job = sampler.run([(transpiled_circuits, param_values)])
result = job.result()
print(result)

