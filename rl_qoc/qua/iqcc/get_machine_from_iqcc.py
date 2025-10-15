from typing import Optional
from iqcc_cloud_client import IQCC_Cloud
import os
import json
from iqcc_calibration_tools.quam_config.components import Quam


def get_machine_from_iqcc(backend_name: str, api_token: Optional[str] = None):
    iqcc = IQCC_Cloud(quantum_computer_backend=backend_name, api_token=api_token)

    # Get the latest state and wiring files
    latest_wiring = iqcc.state.get_latest("wiring")
    latest_state = iqcc.state.get_latest("state")

    # Get the state folder path from environment variable
    quam_state_folder_path = os.environ["QUAM_STATE_PATH"]

    # Save the files
    with open(os.path.join(quam_state_folder_path, "wiring.json"), "w") as f:
        json.dump(latest_wiring.data, f, indent=4)

    with open(os.path.join(quam_state_folder_path, "state.json"), "w") as f:
        json.dump(latest_state.data, f, indent=4)

    machine = Quam.load()

    return machine, iqcc
