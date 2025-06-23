from qm import QuantumMachinesManager, generate_qua_script
from qm.qua import *
from qm_saas.client import QmSaas, QoPVersion
from qm.simulate import SimulationConfig, LoopbackInterface
from pulse_level.qua.configuration import config
from rl_qoc.qua.parameter_table import ParameterTable

# Open communication with the server.
from pathlib import Path
import json

path = Path.home() / "qm_saas_config.json"
with open(path, "r") as f:
    config = json.load(f)
email = config["email"]
password = config["password"]
#
host = "qm-saas.dev.quantum-machines.co"
client = QmSaas(host=host, email=email, password=password)

params = ParameterTable({"param1": 0.5, "param2": 0.7})
with client.simulator(QoPVersion.v2_2_2) as instance:
    qmm = QuantumMachinesManager(
        host=instance.host,
        port=instance.port,
        connection_headers=instance.default_connection_headers,
    )

    with program() as measureProg:
        var = params.declare_variables(False)

        output_stream_1 = declare_stream()
        assign(params["param1"], 0.1)
        play("x", "qubit1$xy")
        save(params["param1"], output_stream_1)

        with stream_processing():
            output_stream_1.save_all("var0")

    print(generate_qua_script(measureProg))
    qm = qmm.open_qm(config=config)
    job = qm.simulate(measureProg, SimulationConfig(int(10000)))
    samples = job.get_simulated_samples()

    result_handles = job.result_handles
    result_handles.wait_for_all_values()
    res = result_handles.get("var0").fetch_all()["value"]


print("Var", res)
