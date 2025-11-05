from iqcc_cloud_client.runtime import get_qm_job
from qiskit.qasm3 import loads
from qiskit.circuit import Parameter
from qiskit.primitives import PrimitiveResult
from qiskit.primitives.containers import DataBin, BitArray
import numpy as np

from qiskit_qm_provider.parameter_table import ParameterTable, InputType

job = get_qm_job()
openqasm_code = ['OPENQASM 3.0;\ninput float[64] param;\nbit[1] meas;\nreset $0;\nrz(1.5707963267948966) $0;\nsx $0;\nrz(3.141592653589793 + param) $0;\nsx $0;\nrz(7.853981633974483) $0;\nbarrier $0;\nmeas[0] = measure $0;\n']
circuits = [loads(code) for code in openqasm_code]
parameter_values = [[[0.        ],
  [0.34906585],
  [0.6981317 ],
  [1.04719755],
  [1.3962634 ],
  [1.74532925],
  [2.0943951 ],
  [2.44346095],
  [2.7925268 ],
  [3.14159265]]]
parameter_tables = [ParameterTable.from_qiskit(circuit, input_type=INPUT_STREAM,
filter_function=lambda x: isinstance(x, Parameter)) for circuit in circuits]
print(parameter_tables)
for parameter_value, parameter_table in zip(parameter_values, parameter_tables):
    if parameter_table is not None:
        param_dict = {param.name: value for param, value in zip(parameter_table.parameters, parameter_value)}
        parameter_table.push_to_opx(param_dict, job)

results_handle = job.result_handles
results_handle.wait_for_all_values()

# all_data = []
# for i, circuit in enumerate(circuits):
#     qc_meas_data = {}
#     for creg in circuit.cregs:
#         data = results_handle.get("" + creg.name + "_" + str(i)).fetch_all()["value"]
#         meas_level = self.metadata.get("meas_level")
#         if meas_level == "classified":
#             bit_array = BitArray.from_samples(data.tolist(), creg.size).reshape(pub.shape)
#             qc_meas_data[creg.name] = bit_array
#         elif meas_level == "kerneled":
#             # TODO: Assume that buffering was done like (2, creg.size)
#             qc_meas_data[creg.name] = np.array([d[0] + 1j * d[1] for d in data], dtype=complex).reshape(
#                 pub.shape + (pub.shots, creg.size)
#             )
#         else:
#             # TODO: Figure it out
#             qc_meas_data[creg.name] = np.array([d[0] + 1j * d[1] for d in data], dtype=complex).reshape(
#                 pub.shape + (pub.shots, creg.size)
#             )

#     sampler_data = SamplerPubResult(DataBin(**qc_meas_data))
#     all_data.append(sampler_data)

# result = PrimitiveResult(all_data)
# print(result)