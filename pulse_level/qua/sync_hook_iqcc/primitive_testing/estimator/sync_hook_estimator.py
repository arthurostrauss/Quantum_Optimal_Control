from iqcc_cloud_client.runtime import get_qm_job
from qiskit_qm_provider.parameter_table import ParameterTable, InputType, Parameter as QMParameter, Direction
from qm.qua import fixed

job = get_qm_job()

parameter_values_list = [[np.array([0.]), np.array([0.6981317]), np.array([1.3962634]), np.array([2.0943951]), np.array([2.7925268]), np.array([3.4906585]), np.array([4.1887902]), np.array([4.88692191]), np.array([5.58505361]), np.array([6.28318531])]]
obs_indices_list = [[[(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)], [(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)], [(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)], [(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)], [(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)], [(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)], [(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)], [(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)], [(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)], [(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]]]
param_tables = [ParameterTable(parameters_dict=[QMParameter(name='param', value=0.0, qua_type=fixed, input_type=None, direction=None, units="")], name='param_table_circuit-41')]
observables_vars = [ParameterTable(parameters_dict=[QMParameter(name='obs_0', value=0, qua_type=int, input_type=None, direction=None, units=""), QMParameter(name='obs_1', value=0, qua_type=int, input_type=None, direction=None, units=""), QMParameter(name='obs_2', value=0, qua_type=int, input_type=None, direction=None, units=""), QMParameter(name='obs_3', value=0, qua_type=int, input_type=None, direction=None, units=""), QMParameter(name='obs_4', value=0, qua_type=int, input_type=None, direction=None, units=""), QMParameter(name='obs_5', value=0, qua_type=int, input_type=None, direction=None, units=""), QMParameter(name='obs_6', value=0, qua_type=int, input_type=None, direction=None, units=""), QMParameter(name='obs_7', value=0, qua_type=int, input_type=None, direction=None, units=""), QMParameter(name='obs_8', value=0, qua_type=int, input_type=None, direction=None, units=""), QMParameter(name='obs_9', value=0, qua_type=int, input_type=None, direction=None, units=""), QMParameter(name='obs_10', value=0, qua_type=int, input_type=None, direction=None, units=""), QMParameter(name='obs_11', value=0, qua_type=int, input_type=None, direction=None, units=""), QMParameter(name='obs_12', value=0, qua_type=int, input_type=None, direction=None, units=""), QMParameter(name='obs_13', value=0, qua_type=int, input_type=None, direction=None, units=""), QMParameter(name='obs_14', value=0, qua_type=int, input_type=None, direction=None, units=""), QMParameter(name='obs_15', value=0, qua_type=int, input_type=None, direction=None, units=""), QMParameter(name='obs_16', value=0, qua_type=int, input_type=None, direction=None, units=""), QMParameter(name='obs_17', value=0, qua_type=int, input_type=None, direction=None, units="")], name='observables_var_circuit-41')]

for i in range(len(parameter_values_list)):
    param_table = param_tables[i]
    observables_var = observables_vars[i]
    parameter_values = parameter_values_list[i]
    obs_indices = obs_indices_list[i]
    
    # Push parameter values if the circuit has them
    if param_table is not None and param_table.input_type is not None:
        for p, param_value in enumerate(parameter_values):
            param_dict = {param.name: value for param, value in zip(param_table.parameters, param_value)}
            param_table.push_to_opx(param_dict, job)
            if observables_var.input_type is not None:
                for obs_value in obs_indices[p]:
                    obs_dict = {f"obs_{q}": val for q, val in enumerate(obs_value)}
                    observables_var.push_to_opx(obs_dict, job)
    # Push observable indices
    elif observables_var.input_type is not None:
        for obs_value in obs_indices[0]:
            obs_dict = {f"obs_{q}": val for q, val in enumerate(obs_value)}
            observables_var.push_to_opx(obs_dict, job)

# results_handle = job.result_handles
# results_handle.wait_for_all_values()

# Post-processing of results (commented out - handled on client side)
# Note: The following code structure shows how results would be processed,
# but requires execution_plans and methods from QMEstimatorJob which are not
# available in the sync hook context. Actual post-processing is done in
# IQCCEstimatorJob._result_function() on the client side.
# 
# pub_results = []
# for i in range(len(parameter_values_list)):
#     # Get bitstring data from result handles
#     # data = np.array(results_handle.get(f"__c_{i}")).flatten().tolist()
#     
#     # Reshape data: (total_tasks * shots, num_bits) -> (total_tasks, shots, num_bits)
#     # bitstrings = np.array(data).reshape(total_tasks, shots, num_qubits)
#     
#     # Compute expectation values from bitstrings
#     # (Requires _calc_expval_map method from QMEstimatorJob)
#     
#     # Postprocess to get PubResult with expectation values and standard errors
#     # (Requires _postprocess_pub method from QMEstimatorJob)
#     # pub_results.append(pub_result)
# 
# # result = PrimitiveResult(pub_results, metadata={"version": 2})
# # print(result)
