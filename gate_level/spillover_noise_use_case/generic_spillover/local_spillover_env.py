from qiskit.transpiler import Layout

from rl_qoc import GateTarget
from rl_qoc.environment import ContextAwareQuantumEnvironment
from rl_qoc.helpers import causal_cone_circuit
from spillover_effect_on_subsystem import noisy_backend


class LocalSpilloverEnvironment(ContextAwareQuantumEnvironment):
    pass
    # def define_target_and_circuits(self):
    #     """
    #     Define the target gate and the circuits to be executed
    #     """
    #     circuit_context = causal_cone_circuit(
    #         self.circuit_context, list(self.config.env_metadata["target_subsystem"])
    #     )[0]
    #     self._physical_target_qubits = list(range(circuit_context.num_qubits))
    #     self._circuit_context = circuit_context
    #     target, custom_circuits, baseline_circuits = (
    #         super().define_target_and_circuits()
    #     )
    #
    #     return target, custom_circuits, baseline_circuits
