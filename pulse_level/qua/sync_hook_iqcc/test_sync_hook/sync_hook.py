from iqcc_cloud_client.runtime import get_qm_job
from qm.qua import *

job = get_qm_job()
result = job.result_handles

optimization_sequence = [0.1, 0.4, 0.06]

for id, value in enumerate(optimization_sequence):
    job.push_to_input_stream("gate_input_stream", value)
    result.measurements.wait_for_values(id + 1)
    print(f"{id}: Received ", str(result.measurements.fetch(id)))

# finish QUA program
job.push_to_input_stream("gate_input_stream", 0)
