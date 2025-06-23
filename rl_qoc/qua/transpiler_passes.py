from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.transpiler.basepasses import AnalysisPass, TransformationPass
from qualang_tools.video_mode import ParameterTable


class QuaParameterTablePass(AnalysisPass):
    """
    This pass is used to generate a ParameterTable object from the parameters of a list of circuits and
    adds it to the property set of the PassManager.
    """

    def __init__(self):
        super().__init__()
        self.parameter_tables = []

    def run(self, dag):
        # Get the parameters from the circuit.
        parameters = dag.parameters
        param_dict = {}
        if parameters:

            # If the circuit has parameters, convert them to a dictionary.
            for parameter in parameters:
                if (
                    isinstance(parameter, ParameterVectorElement)
                    and parameter.vector.name not in param_dict
                ):
                    param_dict[parameter.vector.name] = [0.0 for _ in range(len(parameter.vector))]
                elif isinstance(parameter, Parameter):
                    param_dict[parameter.name] = 0.0

            # Create a ParameterTable object.
            param_table = ParameterTable(param_dict)
            self.parameter_tables.append(param_table)
            # Add the ParameterTable object to the property set
            self.property_set["parameter_table"] = self.parameter_tables

        return dag
