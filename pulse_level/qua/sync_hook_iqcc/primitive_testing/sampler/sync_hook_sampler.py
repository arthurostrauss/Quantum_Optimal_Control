from iqcc_cloud_client.runtime import get_qm_job
from qiskit_qm_provider.parameter_table import ParameterTable, InputType, Parameter as QMParameter, Direction
from qm.qua import fixed

job = get_qm_job()

parameter_values = [[[0.        ],
  [0.03173326],
  [0.06346652],
  [0.09519978],
  [0.12693304],
  [0.1586663 ],
  [0.19039955],
  [0.22213281],
  [0.25386607],
  [0.28559933],
  [0.31733259],
  [0.34906585],
  [0.38079911],
  [0.41253237],
  [0.44426563],
  [0.47599889],
  [0.50773215],
  [0.53946541],
  [0.57119866],
  [0.60293192],
  [0.63466518],
  [0.66639844],
  [0.6981317 ],
  [0.72986496],
  [0.76159822],
  [0.79333148],
  [0.82506474],
  [0.856798  ],
  [0.88853126],
  [0.92026451],
  [0.95199777],
  [0.98373103],
  [1.01546429],
  [1.04719755],
  [1.07893081],
  [1.11066407],
  [1.14239733],
  [1.17413059],
  [1.20586385],
  [1.23759711],
  [1.26933037],
  [1.30106362],
  [1.33279688],
  [1.36453014],
  [1.3962634 ],
  [1.42799666],
  [1.45972992],
  [1.49146318],
  [1.52319644],
  [1.5549297 ],
  [1.58666296],
  [1.61839622],
  [1.65012947],
  [1.68186273],
  [1.71359599],
  [1.74532925],
  [1.77706251],
  [1.80879577],
  [1.84052903],
  [1.87226229],
  [1.90399555],
  [1.93572881],
  [1.96746207],
  [1.99919533],
  [2.03092858],
  [2.06266184],
  [2.0943951 ],
  [2.12612836],
  [2.15786162],
  [2.18959488],
  [2.22132814],
  [2.2530614 ],
  [2.28479466],
  [2.31652792],
  [2.34826118],
  [2.37999443],
  [2.41172769],
  [2.44346095],
  [2.47519421],
  [2.50692747],
  [2.53866073],
  [2.57039399],
  [2.60212725],
  [2.63386051],
  [2.66559377],
  [2.69732703],
  [2.72906028],
  [2.76079354],
  [2.7925268 ],
  [2.82426006],
  [2.85599332],
  [2.88772658],
  [2.91945984],
  [2.9511931 ],
  [2.98292636],
  [3.01465962],
  [3.04639288],
  [3.07812614],
  [3.10985939],
  [3.14159265]]]
parameter_tables = [ParameterTable(parameters_dict=[QMParameter(name='param', value=None, qua_type=fixed, input_type=InputType.INPUT_STREAM, direction=None, units="")], name='param_table_0')]

for parameter_value, parameter_table in zip(parameter_values, parameter_tables):
    if parameter_table is not None and parameter_table.input_type is not None:
        param_dict = {param.name: value for param, value in zip(parameter_table.parameters, parameter_value)}
        parameter_table.push_to_opx(param_dict, job)

# results_handle = job.result_handles
# results_handle.wait_for_all_values()

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