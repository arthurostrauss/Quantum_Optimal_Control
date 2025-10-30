
from rl_qoc import QuantumEnvironment, QiskitConfig, QEnvConfig, ExecutionConfig, StateTarget, ShadowReward, GateTarget
from qiskit.circuit import QuantumCircuit, ParameterVector, QuantumRegister
from qiskit.circuit.library import RXGate, UGate, RZXGate
from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix, random_statevector, Choi, Operator, SuperOp, SparsePauliOp, Pauli

from gymnasium.spaces import Box
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime

""" Shadow Bound Calculation; Taken from Pennylane: Classical Shadows."""
def shadow_bound_state(error, observables, coeffs, failure_rate=0.01):
   
    M = len(observables)
    K = 2 * np.log(2 * M / failure_rate)
    shadow_norm = (
        lambda op: np.linalg.norm(
            op - np.trace(op) / 2 ** int(np.log2(op.shape[0])), ord=np.inf
        )
        ** 2
    )
    N = 34 * max(shadow_norm(observables[i]) * coeffs[i]**2 for i in range(len(observables))) / error ** 2
    
    return max(int(np.ceil(N.real * K)), 100), int(K), M           #sometimes N = 0. A limit of 100 is set to prevent this




def x_at_y_linear(x, y, y_target=0.01):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Find indices where y crosses y_target
    indices = np.where(np.diff(np.sign(y - y_target)))[0]

    if len(indices) == 0:
        x_target = 0
        #raise ValueError("y_target is out of the range of y values.")
    else:
        # Take the first crossing for simplicity
        idx = indices[0]

        # Linear interpolation to find the corresponding x value
        x1, x2 = x[idx], x[idx + 1]
        y1, y2 = y[idx], y[idx + 1]

        # Calculate the slope
        slope = (y2 - y1) / (x2 - x1)

        # Calculate the x value at y_target
        x_target = x1 + (y_target - y1) / slope

    return x_target
# ______________________________________________________________________________________________________________________________________________



# TEST 0: 1 qubits, 1 parameter only
def test_1_qubits():
# simplified 1 qubit circuit of one parameter
    def apply_parametrized_gate(qc: QuantumCircuit, params: ParameterVector, qr: QuantumRegister, *args, **kwargs):
        qc.ry(params[0], 0)

    # 1 qubit parametrized state of one parameter
    theta = np.pi/8 #generate a random target state; this is the goal we want to obtain
    tgt_state = np.cos(theta/2) * Statevector.from_label('0') + np.sin(theta/2) * Statevector.from_label('1')

    #params = np.array([[theta]])
    params = np.array([[np.random.rand()*np.pi] for i in range(2)]) # for only one parameter in the circuit, over a few batches
    return apply_parametrized_gate, tgt_state, params

# ______________________________________________________________________________________________________________________________________________



# TEST 1: 2 qubits, 1 parameter only
def test_2_qubits_1_param():
# simplified 2 qubit circuit of one parameter
    def apply_parametrized_gate(qc: QuantumCircuit, params: ParameterVector, qr: QuantumRegister, *args, **kwargs):
        #test circuit 1
        qc.ry(params[0], 0)
        qc.cx(0,1)
        
        #test circuit 2
        #qc.h(0)
        #qc.cx(0,1)
        #qc.rz(params[0], 1)

    # 2 qubit parametrized bell state of one parameter
    theta = np.pi #generate a random target state; this is the goal we want to obtain
    tgt_state = (np.cos(theta/2) * Statevector.from_label('00') + np.sin(theta/2) * Statevector.from_label('11'))  

    params =  np.array([[theta]])
    #params = np.array([[np.random.rand()*np.pi] for i in range(10)]) # for only one parameter in the circuit, over a few batches

    return apply_parametrized_gate, tgt_state, params



# ______________________________________________________________________________________________________________________________________________

# TEST 2: 2 qubits, 6 parameters

def test_2_qubits():
    #generic 2 qubit circuit of 6 parameters
    def apply_parametrized_gate(qc: QuantumCircuit, params: ParameterVector, qr: QuantumRegister, *args, **kwargs):
        qc.u(params[0], params[1],params[2], 0)
        qc.u(params[3], params[4],params[5], 1)
        qc.cx(0,1)


    def qc_target(params):
    
        qc = QuantumCircuit(2)
        qc.u(params[0], params[1],params[2], 0)
        qc.u(params[3], params[4],params[5], 1)
        qc.cx(0,1)

        return qc

    #np.random.seed(42)
    tgt_param = np.random.rand(6)*2* np.pi
    tgt_state = Statevector(qc_target(tgt_param))
    
    random_param = np.random.rand(6)*2* np.pi
    params = np.array([random_param for i in range(10)])  #test 10 times the same parameters

    return apply_parametrized_gate, tgt_state, params, tgt_param



# _______________________________________________________________________________________________________________________________________________
#to change if you need to use
# TEST 4: 4 qubits, 12 parameters
def test_4_qubits():
#generic 4 qubit circuit of 12 parameters
    def apply_parametrized_gate(qc: QuantumCircuit, params: ParameterVector, qr: QuantumRegister, *args, **kwargs):
        qc.u(params[0], params[1],params[2], 0)
        qc.u(params[3], params[4],params[5], 1)
        qc.cx(0,1)
        qc.cx(0,2)
        qc.cx(0,3)
        qc.u(params[6], params[7],params[8], 2)
        qc.cx(1,2)
        qc.cx(1,3)
        qc.u(params[9], params[10],params[11], 3)
        qc.cx(2,3)

    # 4 qubit random state
    def qc_target(params):

        qc = QuantumCircuit(4)
        qc.u(params[0], params[1],params[2], 0)
        qc.u(params[3], params[4],params[5], 1)
        qc.cx(0,1)
        qc.cx(0,2)
        qc.cx(0,3)
        qc.u(params[6], params[7],params[8], 2)
        qc.cx(1,2)
        qc.cx(1,3)
        qc.u(params[9], params[10],params[11], 3)
        qc.cx(2,3)

        return qc

    #np.random.seed(42)
    tgt_param = np.random.rand(12)*2* np.pi
    tgt_state = Statevector(qc_target(tgt_param))

    random_param = np.random.rand(12)*2* np.pi
    params = np.array([tgt_param for i in range(5)])  #test 5 times the same parameters

    return apply_parametrized_gate, tgt_state, params, tgt_param



# ______________________________________________________________________________________________________________________________________________
def test_4_qubits_choi():
    def apply_parametrized_gate(qc: QuantumCircuit, params: ParameterVector, qr: QuantumRegister, *args, **kwargs):
        """
        |J(U)> builder where U is a *general* 2-qubit unitary on OUT=(q0,q1).
        Qubit order: [OUT0, OUT1, IN0, IN1].
        params: length 24 (overparameterized but universal for SU(4)).
        """
        if len(params) != 24:
            raise ValueError("params must have length 24")
        
        # Prepare |Phi+> across (IN0,OUT0) and (IN1,OUT1)
        qc.h(2); qc.cx(2, 0)
        qc.h(3); qc.cx(3, 1)

        # --- General SU(4) on OUT=(q0,q1): 3 CNOTs + single-qubit layers ---
        # Layer 0
        qc.u(params[0],  params[1],  params[2],  0)
        qc.u(params[3],  params[4],  params[5],  1)
        qc.cx(0, 1)

        # Layer 1
        qc.u(params[6],  params[7],  params[8],  0)
        qc.u(params[9],  params[10], params[11], 1)
        qc.cx(1, 0)

        # Layer 2
        qc.u(params[12], params[13], params[14], 0)
        qc.u(params[15], params[16], params[17], 1)
        qc.cx(0, 1)

        # Final local layer
        qc.u(params[18], params[19], params[20], 0)
        qc.u(params[21], params[22], params[23], 1)



    def choi_pure_state_circuit_for_unitary_su4(params):
        """
        |J(U)> builder where U is a *general* 2-qubit unitary on OUT=(q0,q1).
        Qubit order: [OUT0, OUT1, IN0, IN1].
        params: length 24 (overparameterized but universal for SU(4)).
        """
        if len(params) != 24:
            raise ValueError("params must have length 24")

        qc = QuantumCircuit(4, name="|J(U)>")

        # Prepare |Phi+> across (IN0,OUT0) and (IN1,OUT1)
        qc.h(2); qc.cx(2, 0)
        qc.h(3); qc.cx(3, 1)

        # --- General SU(4) on OUT=(q0,q1): 3 CNOTs + single-qubit layers ---
        # Layer 0
        qc.u(params[0],  params[1],  params[2],  0)
        qc.u(params[3],  params[4],  params[5],  1)
        qc.cx(0, 1)

        # Layer 1
        qc.u(params[6],  params[7],  params[8],  0)
        qc.u(params[9],  params[10], params[11], 1)
        qc.cx(1, 0)

        # Layer 2
        qc.u(params[12], params[13], params[14], 0)
        qc.u(params[15], params[16], params[17], 1)
        qc.cx(0, 1)

        # Final local layer
        qc.u(params[18], params[19], params[20], 0)
        qc.u(params[21], params[22], params[23], 1)

        return qc

    def qc_target_su4(params):
        """
        Universal 2-qubit ansatz (covers SU(4)):
        [U⊗U] - CX(0,1) - [U⊗U] - CX(1,0) - [U⊗U] - CX(0,1) - [U⊗U]
        params: length 24 (8 single-qubit U's × 3 params each).
        """
        if len(params) != 24:
            raise ValueError("params must have length 24")
        qc = QuantumCircuit(2, name="U_su4")

        # block 0
        qc.u(params[0],  params[1],  params[2],  0)
        qc.u(params[3],  params[4],  params[5],  1)

        qc.cx(0, 1)

        # block 1
        qc.u(params[6],  params[7],  params[8],  0)
        qc.u(params[9],  params[10], params[11], 1)

        qc.cx(1, 0)

        # block 2
        qc.u(params[12], params[13], params[14], 0)
        qc.u(params[15], params[16], params[17], 1)

        qc.cx(0, 1)

        # block 3
        qc.u(params[18], params[19], params[20], 0)
        qc.u(params[21], params[22], params[23], 1)

        return qc

    # method 1 to build choi state
    # tgt_param = np.random.rand(24)*2* np.pi
    # qc_target_ = qc_target_su4(tgt_param)

    # target_choi = Choi(SuperOp(qc_target_))
    # tgt_state = DensityMatrix((target_choi.data)) / 4     #divide by number of qubits to make it normalized

    #method 2 to build choi state. Both methods should give the same result and should be able to reach all gates in SU(4).
    tgt_param = np.random.rand(24)*2* np.pi
    tgt_state = Statevector(choi_pure_state_circuit_for_unitary_su4(tgt_param))

    #random_param = np.random.rand(24)*2* np.pi   #use random parameters if you 1. want the error testing to be more robuse; 2. has different tgt param and param dimensions
    params = np.array([tgt_param for i in range(5)])  #test 5 times the same parameters

    return apply_parametrized_gate, tgt_state, params, tgt_param

#______________________________________________________________________________________________________________________________________________




A_fit_inverse_array = []
B_fit_inverse_array = []
shadow_size_99_fidelity_array = []
shadow_size_99_fidelity_model_array = []
print("Current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

for j in range(1):
    apply_parametrized_gate, tgt_state, params, tgt_param = test_4_qubits_choi() 
    #print("Target Parameters: ",tgt_param)
    backend_config = QiskitConfig(apply_parametrized_gate)
    state_target = StateTarget(tgt_state)
 

    observable_decomp = SparsePauliOp.from_operator(Operator(state_target.dm))
    pauli_coeff = observable_decomp.coeffs   #to also be used in shadow bound
    pauli_str = observable_decomp.paulis
    observables = [Pauli(str).to_matrix() for str in pauli_str]

    reward_error = []
    shadow_sizes = []

    err_val = np.linspace(0.1, 2, 20)
    for error in err_val:
        shadow_size, partition, no_observables = shadow_bound_state(error, observables, pauli_coeff)
        #print("Shadow Size, Partition, Number of Observables: ", shadow_size, partition, no_observables)
        batch_size = len(params)
        execution_config = ExecutionConfig(batch_size=batch_size,
                                        sampling_paulis=shadow_size, 
                                        n_shots=1,
                                        seed=42
                                        )
        reward = ShadowReward()
        env_config = QEnvConfig(state_target, 
                                backend_config=backend_config,
                                execution_config=execution_config,
                                action_space=Box(low=np.array([0 for i in range(len(params[0]))]), high=np.array([2*np.pi for i in range(len(params[0]))]), shape=(len(params[0]),)),
                                reward=reward)

        env = QuantumEnvironment(env_config)

        reward_data = reward.get_reward_data(env.circuit, params, state_target, env_config)
        reward_array = reward.get_reward_with_primitive(reward_data, env.sampler)#, state_target)
        #print("Rewards:", reward_array)

        binded_circuits = [env.circuit.assign_parameters(p) for p in params]
        #print("expected rewards:" , [round(state_fidelity(state_target.dm, Statevector(circ)), 4) for circ in binded_circuits])
        expected_rewards =  [round(state_fidelity(state_target.dm, Statevector(circ)), 4) for circ in binded_circuits]
        reward_error.append(np.mean(np.abs(reward_array - expected_rewards)))
        shadow_sizes.append(shadow_size)
    #print(reward_error)


    # do a curve fit. Since log N = A'*log(epsilon) + B', we invert it to log(epsilon) =  A*log N + B
    def model(x, A, B):
        return 10**B * x**A
    # Fit using scipy curve_fit
    popt, pcov = curve_fit(model, shadow_sizes, reward_error, p0=(1.0, 1.0))  # initial guesses for A, B

    A_fit, B_fit = popt
    # print(f"Fitted A = {A_fit:.4f}, B = {B_fit:.4f}")
    shadow_sizes_fitted = np.linspace(min(shadow_sizes), max(shadow_sizes), 20)
    reward_error_fitted =  model(shadow_sizes_fitted , *popt)
    A_fit_inverse = 1/A_fit
    B_fit_inverse = -B_fit / A_fit
    #print(f"Fitted A' = {A_fit_inverse:.4f}, B' = {B_fit_inverse:.4f}")

    shadow_size_99_fidelity = 10 ** x_at_y_linear(np.log10(shadow_sizes), np.log10(reward_error), y_target=np.log10(0.01))
    shadow_size_99_fidelity_model = model(0.01, A_fit_inverse, B_fit_inverse)

    #print(f"Shadow Size for 99% Fidelity: {shadow_size_99_fidelity:.4f}")
    A_fit_inverse_array.append(float(A_fit_inverse))
    B_fit_inverse_array.append(float(B_fit_inverse))
    shadow_size_99_fidelity_array.append(float(shadow_size_99_fidelity))
    shadow_size_99_fidelity_model_array.append(float(shadow_size_99_fidelity_model))

print("A' values: ", A_fit_inverse_array)
print("B' values: ", B_fit_inverse_array)
print("Shadow Size for 99% Fidelity: ", shadow_size_99_fidelity_array)
print("Shadow Size for 99% Fidelity using Model: ", shadow_size_99_fidelity_model_array)
print("Current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


#plotting
num_qubits = 4
plt.plot(shadow_sizes, reward_error,  label="Simulated Error")
plt.plot(shadow_sizes, err_val,  label="Error Bound")
plt.plot(shadow_sizes_fitted, reward_error_fitted,   label="Simulated Error Fitted")
plt.xlabel("Number of Snapshots")       # Label for X-axis
plt.ylabel("Absolute Error")       # Label for Y-axis
plt.title(f"Error Boundedness Check for {num_qubits} qubits")     # Graph title
plt.legend()                     # Show legend
plt.xscale('log')
plt.yscale('log')
plt.show()

