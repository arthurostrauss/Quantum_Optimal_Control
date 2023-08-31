# Feedback-based Quantum Control using Reinforcement Learning 

This repository is dedicated to the writing of Python scripts enabling the realization of quantum control tasks based on closed-loop optimization that incorporates measurements. We build upon the work of Volodymir Sivak on Model-Free Quantum Control with Reinforcement Learning (https://link.aps.org/doi/10.1103/PhysRevX.12.011059) to enable a generic framework for arbitrary state preparation and quantum gate calibration based on Qiskit modules. The interest here is that this repo enables the execution of the algorithm over both simulation and real IBM Backends seamlessly, and therefore enables an in-depth analysis of the robustness of the method for a variety of different backends. In the future, we will add compatibility with further providers. Additionally, this repository is dedicated to the investigation of contextual gate calibration.

The repo is currently divided in five main folders. The first one is the folder "paper_results", where adaptation of the state preparation algorithm developed by Sivak for bosonic systems is adapted to a multiqubit system simulated with a Qiskit ``` QuantumCircuit``` running on a ```QasmSimulator``` backend instance.

The two other folders extend the state preparation protocols by integrating quantum gate calibration procedures, that can be described using a usual gate level abstraction (that is one seeks to optimize gate parameters, in the spirit of a variational quantum algorithm), or at the pulse level abstraction, for more expressivity and harware-aware noise suppression. The novelty of this gate calibration procedure is that it can be tailormade to a quantum circuit context, enabling the suppression of spatially and temporally correlated noise.

We briefly explain the overall logic of the algorithm below.

## Introduction

### Reinforcement Learning 101
Reinforcement Learning (RL) is an interaction-based learning procedure resulting from the interplay between two entities:
- an Environment: in our case the quantum system of interest formed of $n$ qubits, characterized by a quantum state $\rho\in \mathcal{H}$, where the dimension of Hilbert space $\mathcal{H}$ is $d=2^n$ and from which 
observations/measurements $o_i \in \mathcal{O}$  can be extracted. Measurements could be retrieved as bitstrings ($\mathcal{O}= \{0, 1\}^{\otimes n}$) or potentially as IQ pairs if state discrimination is not done, as shown in this recent work (https://arxiv.org/abs/2305.01169). The point here is that we restrict the observation space to values that can be retrieved from the quantum device. Usually, we do not have direct access to typical quantum control benchmarks such as state or gate fidelity, making the learning procedure more difficult as our quantum environment (in the RL sense) is partially observable.
- an Agent: a classical Neural Network on the PC, whose goal is to learn a stochastic policy $\pi_\theta(a|o_i)$  from which actions $a \in \mathbb{U}$ are sampled to be applied on the Environment to set it in a specific target state. Typically, the goal of the Agent in the frame of Quantum Optimal Control is to find actions (e.g., circuit/pulse parameters) enabling the successful preparation of a target quantum state, or the calibration of a quantum gate operation.

### Context aware gate calibration
Quantum computing is entering a phase where the prospect of obtaining a comparative advantage over classical computing is now foreseeable in the near future. However, one of the main obstacles to the emergence of this technology lies in the fact that quantum processors are extremely sensitive to environmental disturbances. These disturbances lead to errors in quantum information processing, making it difficult to interpret results when executing a quantum algorithm. Numerous research efforts have been made in recent years to limit error rates (e.g., improving the quality of quantum processors, synthesizing logic gates more robust to noise) and correct residual errors (quantum error correction). Although these methods are all promising for enabling the execution of more ambitious algorithms on large-scale quantum processors, they currently do not provide robustness to dynamic and contextual errors. Specifically, we are interested in errors that emerge when a specific quantum circuit is executed on the processor, whose physical origin can only be attributed to the unique sequence of logic gates executed within this circuit. This circuit context may generate errors carrying either temporal (e.g., non-Markovian noise) or spatial dependency (undesired crosstalk between neighboring qubits on the processor). As this spatio-temporal dependency can be difficult to model or effectively characterize on intermediate-sized processors, we propose an algorithm based on model free reinforcement learning to determine the best possible calibration of error-prone quantum gates that are tailor-made for a specific quantum circuit context. 

What we want to achieve here is to show that model-free quantum control with RL (Sivak et. al, https://link.aps.org/doi/10.1103/PhysRevX.12.011059) is able to capture the contextual dependence of such gate calibration task and to provide suitable error suppression strategies. In the examples currently available, we explain the overall logic of the algorithm itself, and detail later how the context dependence can be encompassed through a tailor-made training procedure.

![Programming Pipeline](Programming_pipeline.png) 
## 1. Describing the QuantumEnvironment
### a. Deriving the reward for the quantum control problem
As explained above, our framework builds upon the idea of transforming the quantum control task into a RL problem where an agent is missioned to probe and act upon an environment in order to steer it into a target state. The way this is usually done in RL is to introduce the concept of a reward signal $R$, which should be designed such that the maximization of this reward yields a set of actions providing a successful and reliable target preparation. 
For a quantum state preparation task, this reward should be designed such that:

![equation](https://latex.codecogs.com/svg.image?\inline&space;\mathrm{argmax}_{|\psi\rangle}\,\mathbb{E}_{\pi_\theta}[R]=|\psi_{target}\rangle)

Where the expectation value is empirically taken over a batch of actions sampled from the policy $\pi_\theta$ (Monte Carlo sampling) yielding different rewards. It is clear that the reward should therefore act as a proxy for a distance measure between quantum states such as the fidelity. 
The average reward could therefore be a statistical estimator for the fidelity. The Direct Fidelity Estimation (DFE) scheme introduced in [2] gives us a protocol to build a single shot reward scheme based on Pauli expectation sampling. We take the same notation as in the paper below and write the corresponding protocol to derive the reward.

The ﬁdelity between our desired pure state $\rho$ and the actual state $\sigma$ is given by:
$$F(\rho, \sigma)=\left(\mathrm{tr}\left[(\sqrt{\rho} \sigma \sqrt{\rho})^{1 / 2}\right]\right)^2=\mathrm{tr}(\rho \sigma)$$

Let $W_k \, (k=1,..,d^2)$ denote all possible Pauli operators acting on $n$ qubits, that is all $n$-fold tensor products of Pauli matrices (![equation](https://latex.codecogs.com/svg.image?\\inline&space;\large&space;\dpi{110}I=\begin{pmatrix}1&0\\0&1\end{pmatrix}), ![equation](https://latex.codecogs.com/svg.image?\inline&space;\large&space;\sigma_x=\begin{pmatrix}0&1\\1&0\end{pmatrix}), ![equation](https://latex.codecogs.com/svg.image?\inline&space;\large&space;\sigma_y=\begin{pmatrix}0&-i\\i&0\end{pmatrix}), ![equation](https://latex.codecogs.com/svg.image?\inline&space;\large&space;\sigma_z=\begin{pmatrix}1&1\\1&0\end{pmatrix})).

Define the characteristic function $$\chi_\rho(k)=\mathrm{tr}(\rho W_k/\sqrt{d})=\langle W_k\rangle_\rho/\sqrt{d}$$ 
where the expectation value is evaluated on the state $\rho$.

Since those Pauli operators form an orthogonal basis for the density operators under the Hilbert-Schmidt product, one can see that:
$$F(\rho, \sigma)=\mathrm{tr}(\rho \sigma)=\sum_k\chi_\rho(k)\chi_\sigma(k)=\sum_k\frac{\langle W_k\rangle_\rho\langle W_k\rangle_\sigma}{d}$$.

Note that for our target state $\rho$, we supposedly are able to evaluate analytically the expected value of each Pauli operator $\langle W_k\rangle_\rho$. However, we would have to experimentally sample from the quantum computer to deduce the value of $\langle W_k\rangle_\sigma$.
In the original DFE paper, the idea was to provide an estimation of the required number of samples to reach a certain accuracy on the fidelity estimation. However, in our RL case, we are not interested in systematically getting an accurate estimation, as we would like the agent to rather explore multiple trajectories in parameter space (for one fixed policy) in order to quickly evaluate if the chosen policy should be discarded, or if it should try to fine-tune it to push the reward even further. This fact is commonly known in RL as the exploration/exploitation tradeoff. What we do, in line with the protocol is to build an estimator for the fidelity by selecting a set of random Pauli observables $W_k$ to sample from the quantum computer by choosing $k\in\{1,...,d^2\}$ such that:

$$\mathrm{Pr}(k)=[\chi_\rho(k)]^2$$

Once those indices have been sampled, we compute the expectation values $\langle W_k\rangle_\sigma$ by directly sampling in the appropriate Pauli basis on the quantum computer. We let the user choose how many shots per expectation value shall be executed, as well as the number of indices $k$ to be sampled. The choice of those hyperparameters will have a direct impact on the estimation accuracy of the actual state $\sigma$ properties and can therefore impact the convergence of the algorithm. 
One can show that by constructing the estimator $X = \langle W_k\rangle_\sigma/\langle W_k\rangle_\rho$, it follows that:

$$\mathbb{E}_{k\sim \mathrm{Pr(k)}}[X]=F(\rho,\sigma)$$

We can define a single shot reward $R$ by replacing the actual expectation value $\langle W_k\rangle_\sigma$ by an empirical estimation $\mathbb{E}_\sigma[W_k]$, that is measure a finite number of times the created state $\sigma$ in the $W_k$ basis. In general, only the computational basis measurements are natively available on the quantum computer (specifically true for IBM backends), so additional local qubit rotations are  necessary before performing the measurement. This experimental workflow is automatically taken care of in our algorithm by the use of a Qiskit Estimator primitive (https://qiskit.org/documentation/partners/qiskit_ibm_runtime/tutorials/how-to-getting-started-with-estimator.html).

Our single shot reward $R$ can therefore be defined as: 

$$R=\frac{1}{d}\langle W_k\rangle_\rho W_k/\mathrm{Pr}(k)$$.

It is easy to see that we have:

$$\mathbb{E}_{\pi_\theta, k\sim\mathrm{Pr}(k), \sigma} [R]=F(\rho, \sigma)$$


For a gate calibration task, we want to find parameters defining the quantum gate $G(a)$ that yield a high fidelity measure for all possible input states when comparing to the target $G_{target}$. We can therefore implement the same idea as target state preparation reward, but this time we add an averaging over all possible input states $|\psi_{input}\rangle$ such that for each input state, $G(a)|\psi_{input}\rangle=G_{target}|\psi_{input}\rangle$. The average gate fidelity is therefore given by:
![equation](https://latex.codecogs.com/svg.image?\inline&space;\large&space;\mathbb{E}_{\pi_\theta,\,|\psi_{input}\rangle,\,k\sim\mathrm{Pr}(k),\,\sigma}[R]=F(G,G_{target})).

In the code, each average is empirically done over specific hyperparameters that must be tuned by the user at each experiment:

- For the average over the policy (![equation](https://latex.codecogs.com/svg.image?\inline&space;\large&space;\mathbb{E}_{\pi_\theta[.])): the user should adjust the ```batchsize``` hyperparameter, which indicates how many actions from one policy with fixed parameters should be drawn. 
- For the average over input states: at each episode of learning, we sample a different random input state $|\psi_{input}\rangle$ for which the corresponding target state $|\psi_{target}\rangle= G_{target}|\psi_{input}\rangle$ is calculated. Once the target state is deduced, we can start the episode by applying our parametrized gate and measuring the set of Pauli observables to design the reward circuit (Pauli basis rotation at the end of the circuit).
- For the average over the random Pauli observables to sample: the user should adjust the ```sampling_Pauli_space``` hyperparameter. This number will set the number of times an index $k$ shall be sampled based on the previously defined probability $\mathrm{Pr}(k)$. If the same index is sampled multiple times, this induces that the same Pauli observable will be sampled more accurately as the number of shots for one Pauli observable will scale as "number of times $k$ was sampled $\times N_{shots}$". However, due to a constraint of Qiskit, it is not possible to adjust the number of shots for each individual observable. Moreover, it could be that some observables can be inferred from the same measurements (commuting observables). Due to those facts, we settle for 
- The average over the number of shots to estimate each Pauli observable with a certain accuracy: The user should adjust the parameter ```n_shots```, this will be the minimum number of samples that each observable will be estimated with. 

Finally, the last hyperparameter of the overall RL algorithm is the number of epochs ```n_epochs```, which indicates how many times the policy parameters should be updated to try to reach the optimal near-deterministic policy enabling the successful preparation of the target state/gate.

### b. Modelling the quantum system through Qiskit: the ```QuantumCircuit```workflow

Our repo relies on the modelling of the quantum system as a class ```QuantumEnvironment```, which is characterized by the following attributes:

- ``` target: Dict```: a dictionary containing information about the target that the RL agent should prepare. It could either be a target state or a target gate. This dictionary should contain the following keys:
    - ```gate: qiskit.circuit.Gate```: if the target type is a gate, which quantum gate is the target the agent should calibrate
    - ```circuit: qiskit.circuit.QuantumCircuit```: If target type is a state, provides an ideal quantum circuit that should prepare the target state of interest
    - ```dm: qiskit.quantum_info.DensityMatrix```: Alternatively to the ```circuit```key, one can provide a ```DensityMatrix```describing the target state of interest (if target type is a state).
    - ```register: Union[QuantumRegister, List[int]]```: List of qubits on which the target state/gate shall be prepared. The indices provided in this list should match the number of qubits addressed by the target gate/state.
    - ```input_states: Optional[List[Dict[Union["circuit", "dm"]]: Union[qiskit.circuit.QuantumCircuit, qiskit.quantum_info.DensityMatrix]]]```: If target type is a gate, optional list of possible input states that should be prepared for forcing the agent to perform simultaneous successful state preparation. In order to maximize learning, the set of input states should be tomographically complete. Each input state should be provided as if it was a target state (either provide a quantum circuit enabling its perfect preparation, or a DensityMatrix). Note that it does not need to be provided, and the default will load a tomographically complete set.
- ```abstraction_level: str = Union["gate", "pulse"]```: The abstraction level at which the parametrized circuit enabling the target preparation is expressed. If the gate level is chosen, then the simulator that is used is a Statevector simulator, and significantly speeds up the simulation, unless another ```AerBackend``` object is provided in the ```Qiskit_config```below. Contrariwise, if the pulse level is chosen, all gates provided in the parametrized circuit should contain a dedicated pulse description, and the simulation of the pulse schedule associated to the circuit is done through Qiskit Dynamics module (https://qiskit.org/documentation/dynamics). 
- ```Qiskit_config: QiskitConfig```: A dataclass object containing necessary information to execute the algorithm on real/simulated hardware. It should contain the following keys:
    - ```backend: Optional[qiskit.providers.Backend]```: An IBM Backend on which should be executed the overall workflow. If a real backend is selected, then the user should ensure that he has opened an IBM account to access the resource, and that this backend is selected among the ones enabling the execution of Qiskit Runtime (see code examples about how to declare such real backend). If the ```abstraction_level```is set to ```"gate"```, and the user wants to run on a simulated noiseless backend, the ```backend``` can be set to ```None```. Alternatively, the user can provide a custom ```AerBackend``` with a custom noise model. Finally, if the user wants to run the workflow on a simulated backend at the pulse level, a ```qiskit_dynamics.DynamicsBackend``` instance should be provided (see pulse level abstraction folder).
    - ```parametrized_circuit: Callable[qiskit.circuit.QuantumCircuit, Optional[qiskit.circuit.ParameterVector], Optional[qiskit.circuit.QuantumRegister]] ```: A function taking as input a ```QuantumCircuit```instance and appending to it a custom parametrized gate (specified either as a pulse schedule or a parametrized gate depending on the abstraction level). This is the key function that should introduce circuit parameters (```qiskit.circuit.Parameter``` or ```qiskit.circuit.ParameterVector```) which will be replaced by the set of actions that the agent will apply on the system.
    - Other optional arguments can be provided if a real backend or pulse level abstraction are selected. They can be found in the notebook contained in the pulse level abstraction folder.
- ```sampling_Pauli_space: int```: As mentioned earlier, indicates how many indices should be sampled for estimating the fidelity from Pauli observables.
- ```n_shots: int```: Minimum number of shots per expectation value sampling subroutine.
- ```c_factor: float```: Normalization factor for the reward, should be chosen such that average reward curve matches average gate fidelity curve (if target type is a gate) or state fidelity curve (if target type is a state).


### c. The subtlety of the pulse level: Integrating Qiskit Dynamics and Qiskit Experiments

While a ```QuantumCircuit``` can be simulated using Statevector simulation when only dealing with the circuit level abstraction, and does not necessarily require an actual
backend (although one might want to use the ```ibmq_qasm_simulator```), simulating the circuit by providing a pulse level description requires a dedicated simulation backend.
As our main code revolves around Qiskit structure, we use the recently released Qiskit Dynamics extension, which is a framework for simulating arbitrary pulse schedules.
More specifically, we leverage the introduction of the ```DynamicsBackend``` object to emulate a real backend, receiving Pulse gates (you can learn more by checking Qiskit and Dynamics documentations).
The DynamicsBackend can be declared in two main ways:
1. Using the ```from_backend(BackendV1)``` method, which creates a ```DynamicsBackend``` carrying all the properties (in particular the Hamiltonian) of a real IBM backend.
2. Creating a custom Hamiltonian (or Linbladian) describing our quantum system of interest, and declare a ```Solver``` object which will be provided to the backend.

We provide examples of both declarations in the pulse_level_abstraction folder.

The subtlety behind this ```DynamicsBackend``` is that this backend does not carry calibrations for elementary quantum gates on the qubits.
This is an issue, as the current reward scheme is built on the ability to perform Pauli expectation sampling on the quantum computer.
Indeed, since we only have access to a Z-basis measurement in the backend, one needs to be able to perform Pauli basis rotation gates in order to ensure we can extract all Pauli operators. As a consequence, each qubit should have at the start calibrated Hadamard and S gates. The whole idea is to provide such baseline calibration for the
custom ```DynamicsBackend``` instance. To do so, we use the library of elementary calibrations provided in the Qiskit Experiments module (which provide calibration workflows for elementary gates).
Therefore, each time the ```QuantumEnvironment``` object is initialized with a ```DynamicsBackend``` object, a series of baseline calibrations will automatically start, such that the Estimator primitive (in this case the ```BackendEstimator``` present in Qiskit primitives)
can append the appropriate gates for all Pauli expectation value sampling tasks.

## 2. Integrating context awareness to the Environment

In this repo, there are mainly two RL workflows that are used: Tensorflow 2 and PyTorch. While Tensorflow is used for the earlier examples of state preparation and gate calibration, we have for now fixed the context aware calibration workflow on PyTorch for more clarity. More specifically, the whole RL workflow is based on a "Gymified" ```QuantumEnvironment``` called ```TorchEnvironment``` and an Agent built on Torch modules. The implementation of the PPO follows closely the one proposed by the CleanRL library (http://docs.cleanrl.dev/).

In what follows, we explain the logic behind the context-aware gate calibration workflow.

### a. the ```TorchQuantumEnvironment``` wrapper

The main motivation behind this work comes from the realization that in most traditional gate calibration procedures, the noise that could emerge from a specific sequence of logical operations is often neglected or averaged out over a large series of different sequences to build an "on-average" noise robust calibration of the target gate.
While this paradigm inherent to the gate model modular structure is certainly convenient for the execution of complex quantum circuits, we have to acknowledge that this might not be enough to reach satisfying circuit fidelities in the near-term. In fact, spatio-temporally correlated noise sources such as non-Markovianity or crosstalk could impact significantly the quality of a typical gate. It is to be noted that such effect is typically dependent on the operations that are executed around the target gate. 
Characterizing this noise is known to be tedious because of the computational heaviness of the process (e.g. Simultaneous RB for crosstalk, Process tensor tomography for non-Markovianity). In typical experiments, we first spend some time characterizing the noise channels, and then derive a series of gate calibrations countering all the effects simultaneously. 
In our quantum control approach, we decide to intertwine the characterization and the calibration procedure, by letting the RL agent build from its training an internal representation of the contextual noise concurrently to the trial of new controls for suppressing it.

Earlier, we have defined our ```QuantumEnvironment``` to include all details regarding the reward computation and the quantum system access (simulation or real backend). In this new ```TorchQuantumEnvironment``` class, we include all the details regarding the calibration done within a circuit context training.

For this, we first declare the ```TorchQuantumEnvironment``` as follows:




## 3. The RL Agent: PPO algorithm






## References
[1] V. V. Sivak, A. Eickbusch, H. Liu, B. Royer, I. Tsioutsios, and M. H. Devoret, “Model Free Quantum Control with Reinforcement Learning”, Physical Review X, vol. 12, no. 1, p. 011 059, Mar. 2022

[2] S. T. Flammia and Y.-K. Liu, “Direct Fidelity Estimation from Few Pauli Measurements”, Physical Review Letters, vol. 106, no. 23, p. 230 501, Jun. 2011, Publisher: American Physical Society. DOI: 10.1103/PhysRevLett.106.230501. [Online]. Available: https://link.aps.org/doi/10.1103/PhysRevLett.106.230501
