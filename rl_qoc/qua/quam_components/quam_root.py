import os
from pathlib import Path
from quam.core import QuamRoot, quam_dataclass
from quam.components.octave import Octave
from quam.components.ports import (
    LFFEMAnalogOutputPort,
    LFFEMAnalogInputPort,
    FEMDigitalOutputPort,
    OPXPlusAnalogOutputPort,
    OPXPlusAnalogInputPort,
    OPXPlusDigitalOutputPort,
    FEMPortsContainer,
    OPXPlusPortsContainer,
)
from .transmon import Transmon
from .transmon_pair import TransmonPair

from qm.qua import align
from qm import QuantumMachinesManager, QuantumMachine
from qualang_tools.results.data_handler import DataHandler

from dataclasses import field
from typing import List, Dict, ClassVar, Any, Sequence, Union


__all__ = ["QuAM", "FEMQuAM", "OPXPlusQuAM"]


@quam_dataclass
class QuAM(QuamRoot):
    """Example QuAM root component."""

    octaves: Dict[str, Octave] = field(default_factory=dict)

    qubits: Dict[str, Transmon] = field(default_factory=dict)
    qubit_pairs: List[TransmonPair] = field(default_factory=list)
    wiring: dict = field(default_factory=dict)
    network: dict = field(default_factory=dict)

    active_qubit_names: List[str] = field(default_factory=list)
    active_qubit_pair_names: List[str] = field(default_factory=list)

    _data_handler: ClassVar[DataHandler] = None

    @classmethod
    def load(cls, *args, **kwargs) -> "QuAM":
        if not args:
            if "QUAM_STATE_PATH" in os.environ:
                args = (os.environ["QUAM_STATE_PATH"],)
            else:
                raise ValueError(
                    "No path argument provided to load the QuAM state. "
                    "Please provide a path or set the 'QUAM_STATE_PATH' environment variable. "
                    "See the README for instructions."
                )
        return super().load(*args, **kwargs)

    def save(
        self,
        path: Union[Path, str] = None,
        content_mapping: Dict[str, str] = None,
        include_defaults: bool = False,
        ignore: Sequence[str] = None,
    ):
        if path is None and "QUAM_STATE_PATH" in os.environ:
            path = os.environ["QUAM_STATE_PATH"]

        super().save(path, content_mapping, include_defaults, ignore)

    @property
    def data_handler(self) -> DataHandler:
        """Return the existing data handler or open a new one to conveniently handle data saving."""
        if self._data_handler is None:
            self._data_handler = DataHandler(
                root_data_folder=self.network["data_folder"]
            )
            DataHandler.node_data = {"quam": "./state.json"}
        return self._data_handler

    @property
    def active_qubits(self) -> List[Transmon]:
        """Return the list of active qubits."""
        return [self.qubits[q] for q in self.active_qubit_names]

    @property
    def active_qubit_pairs(self) -> List[TransmonPair]:
        """Return the list of active qubits."""
        return self.qubit_pairs

    @property
    def depletion_time(self) -> int:
        """Return the longest depletion time amongst the active qubits."""
        return max(q.resonator.depletion_time for q in self.active_qubits)

    @property
    def thermalization_time(self) -> int:
        """Return the longest thermalization time amongst the active qubits."""
        return max(q.thermalization_time for q in self.active_qubits)

    def apply_all_couplers_to_min(self) -> None:
        """Apply the offsets that bring all the active qubit pairs to a decoupled point."""
        align()
        for qp in self.active_qubit_pairs:
            qp.coupler.to_decouple_idle()
        align()

    def apply_all_flux_to_min(self) -> None:
        """Apply the offsets that bring all the active qubits to the minimum frequency point."""
        align()
        for q in self.active_qubits:
            q.z.to_min()
        align()

    def apply_all_flux_to_zero(self) -> None:
        """Apply the offsets that bring all the active qubits to the zero bias point."""
        align()
        for q in self.active_qubits:
            q.z.to_zero()
        align()

    def connect(self) -> QuantumMachinesManager:
        """Open a Quantum Machine Manager with the credentials ("host" and "cluster_name") as defined in the network file.

        Returns: the opened Quantum Machine Manager.
        """
        settings = dict(
            host=self.network["host"],
            cluster_name=self.network["cluster_name"],
            octave=self.get_octave_config(),
        )
        if "port" in self.network:
            settings["port"] = self.network["port"]
        return QuantumMachinesManager(**settings)

    def get_octave_config(self) -> dict:
        """Return the Octave configuration."""
        octave_config = None
        for octave in self.octaves.values():
            if octave_config is None:
                octave_config = octave.get_octave_config()
            else:
                octave_config.add_device_info(octave.name, octave.ip, octave.port)

        return octave_config

    def calibrate_octave_ports(self, QM: QuantumMachine) -> None:
        """Calibrate the Octave ports for all the active qubits.

        Args:
            QM (QuantumMachine): the running quantum machine.
        """
        from qm.octave.octave_mixer_calibration import NoCalibrationElements

        for name in self.active_qubit_names:
            try:
                self.qubits[name].calibrate_octave(QM)
            except NoCalibrationElements:
                print(
                    f"No calibration elements found for {name}. Skipping calibration."
                )


@quam_dataclass
class FEMQuAM(QuAM):
    ports: FEMPortsContainer = field(default_factory=FEMPortsContainer)


@quam_dataclass
class OPXPlusQuAM(QuAM):
    ports: OPXPlusPortsContainer = field(default_factory=OPXPlusPortsContainer)
