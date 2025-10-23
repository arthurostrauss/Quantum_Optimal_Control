# Feedback-based Quantum Control using Reinforcement Learning 

This repository provides a comprehensive framework for feedback-based quantum control using reinforcement learning. It builds upon the foundational work of Volodymir Sivak on Model-Free Quantum Control to offer a versatile toolkit for tasks like arbitrary state preparation and quantum gate calibration.

The framework has two primary focuses:

1.  **Full Support for Qiskit Primitives**: We provide complete integration with the Qiskit primitives model, allowing for a high-level, hardware-agnostic approach to defining and executing RL-based quantum control workflows. This enables seamless execution on both simulated and real IBM backends, facilitating in-depth analysis of the method's robustness across different platforms.

2.  **Advanced Real-Time Embedded RL with Quantum Machines**: For advanced users, we offer support for real-time embedded reinforcement learning on the Quantum Orchestration Platform (QOP) from Quantum Machines. This powerful feature significantly reduces execution overhead by enabling:
    *   **On-the-fly action sampling**: Actions are sampled and applied in real-time, eliminating the need for pre-compiled sequences.
    *   **Real-time parameter adjustment**: The agent can adjust parameters on the fly, responding dynamically to the system's behavior.
    *   **Zero compilation latency**: By leveraging real-time control flow (like for-loops and switch statements), we eliminate compilation latency, leading to massive speedups. The core strategy relies on the ability to run real-time control flow and real-time parameter adjustment, while maintaining a highly flexible communication workflow between classical and quantum resources for making the program fully parametric and dynamic.

This repository is a powerful tool for researchers and developers interested in exploring the frontiers of quantum control, from high-level, context-aware gate calibration to real-time, hardware-embedded reinforcement learning. The folder `paper_results` contains an adaptation of the state preparation algorithm for a multiqubit system, while other folders extend this to gate calibration at both the gate and pulse level, with a focus on context-aware calibration to suppress correlated noise.

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

Note that for our target state $\rho$, we supposedly are able to evaluate analytically the expected value of each Pauli operator $\langle W_k\rangle_\rho$. However, we would have to experimentally sample from the quantum computer to deduce the value of $\langle W_k\rangle_\sigma`.
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

Our repo relies on the modelling of the quantum system as a class `QuantumEnvironment`, which is characterized by a dataclass `QEnvConfig`. The config contains the following attributes:

- `target`: A `GateTarget` or `StateTarget` object containing information about the target that the RL agent should prepare.
- `backend_config`: A dataclass object (`BackendConfig`) with the necessary information to execute the algorithm on real or simulated hardware. It includes:
    - `backend`: An optional Qiskit `Backend` object. If not provided, a noiseless statevector simulator is used.
    - `parametrized_circuit`: A callable that adds a custom parametrized gate to a `QuantumCircuit`. This is where you define the actions the agent can take.
- `execution_config`: A dataclass object (`ExecutionConfig`) that specifies execution parameters, such as:
    - `n_shots`: The number of shots per expectation value sampling.
    - `sampling_paulis`: The number of Pauli observables to sample for fidelity estimation.
    - `batchsize`: The number of actions to sample from the policy for each update.
    - `n_reps`: The ability to repeat a circuit multiple times to amplify sensitivity to small noise.
- `action_space`: The Gym `Box` space defining the range of possible actions.
- `benchmark_config`: An optional dataclass object (`BenchmarkConfig`) for specifying benchmarking parameters.

### c. The subtlety of the pulse level: Integrating Qiskit Dynamics and Qiskit Experiments

Simulating at the pulse level requires a dedicated simulation backend. We use Qiskit Dynamics and its `DynamicsBackend` to emulate a real backend. This can be created from a `FakeBackend` or a custom Hamiltonian.

A key subtlety is that `DynamicsBackend` does not come with pre-calibrated elementary gates. Since our reward scheme relies on Pauli basis rotations (Hadamard and S gates), we need these calibrations. Our framework automatically handles this by using Qiskit Experiments to perform baseline calibrations for X and SX gates whenever a `DynamicsBackend` is initialized. This ensures that the `BackendEstimator` can perform the necessary Pauli expectation value sampling.

## 2. Integrating context awareness to the Environment

The context-aware gate calibration workflow is built on a "Gymified" `QuantumEnvironment` called `ContextAwareQuantumEnvironment` and an Agent built on PyTorch. The PPO implementation follows the CleanRL library.

### a. the `ContextAwareQuantumEnvironment` wrapper

Traditional gate calibration often averages out noise from specific gate sequences. However, spatio-temporal noise like non-Markovianity and crosstalk can significantly impact gate quality depending on the circuit context.

Our `ContextAwareQuantumEnvironment` addresses this by intertwining noise characterization and calibration. The RL agent learns an internal representation of the contextual noise while simultaneously trying new controls to suppress it.

To use it, you first declare the `ContextAwareQuantumEnvironment` as follows:

```python
from rl_qoc import ContextAwareQuantumEnvironment, QEnvConfig, ExecutionConfig, GateTarget, BackendConfig
from gymnasium.spaces import Box

# Define the configuration for the environment
q_env_config = QEnvConfig(
    target=GateTarget(gate=ECRGate(), physical_qubits=[0, 1]),
    backend_config=BackendConfig(
        parametrized_circuit=apply_parametrized_circuit,
        backend=backend,
    ),
    execution_config=ExecutionConfig(batchsize=300, sampling_paulis=50, n_shots=200),
    action_space=Box(low=-0.1, high=0.1, shape=(4,)),
)

# Define the circuit context and training steps per gate
env = ContextAwareQuantumEnvironment(
    q_env_config,
    circuit_context=transpiled_circ,
)
```

## 3. The RL Agent: PPO algorithm






## References
[1] V. V. Sivak, A. Eickbusch, H. Liu, B. Royer, I. Tsioutsios, and M. H. Devoret, “Model Free Quantum Control with Reinforcement Learning”, Physical Review X, vol. 12, no. 1, p. 011 059, Mar. 2022

[2] S. T. Flammia and Y.-K. Liu, “Direct Fidelity Estimation from Few Pauli Measurements”, Physical Review Letters, vol. 106, no. 23, p. 230 501, Jun. 2011, Publisher: American Physical Society. DOI: 10.1103/PhysRevLett.106.230501. [Online]. Available: https://link.aps.org/doi/10.1103/PhysRevLett.106.230501

## Installation of optional dependencies

Some optional dependencies, especially for integration with other hardware or software platforms, are available as extras in the `pyproject.toml` file:

- `qibo`: qibo, qibolab, qibocal
- `qua`: qualang-tools, qm-qua, quam, quam-builder
- `qiskit-pulse`: qiskit-dynamics, qiskit-experiments

To install an extra, for example:

```bash
pip install .[qua]
```

### Private dependencies

Two optional dependencies, `qiskit-qm-provider` and `oqc`, are private GitHub repositories and cannot be installed automatically. If you have access, install them manually:

```bash
pip install git+ssh://git@github.com:YOUR_ORG/qiskit-qm-provider.git
pip install git+ssh://git@github.com:YOUR_ORG/oqc.git
```

Contact the maintainers to obtain access to these repositories if needed.
