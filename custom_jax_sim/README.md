# Running the training with optimized simulations through Jax.

As you may have seen in the other available examples, the code structure is the same
regardless of the choice of the backend. Whether it's a real quantum backend or a circuit/pulse level simulator,
Qiskit modules enable us to run the same code seamlessly on both thanks to a nice encapsulation.

However, there's one specific use case preventing us of really taking advantage of the full power of fast computation,
the one where we use pulse level simulation for our typical use case.

In fact, the main limitation is related to the current way Qiskit Dynamics leverages Jax to run fast simulations.
While this tool clearly enables faster simulations than simple numpy, we face significant running times when trying to
simulate an important batch of quantum circuits (in our case a single parametrized quantum circuit evaluated over a
batch of different parameter values). The reason for such computation times is that the Estimator primitive we're using
never takes advantage of the fact that we have one single circuit that just needs to be fed different parameters, which
could be straightforwardly translated to pulse parameters (either through automatic transpilation and scheduling, or
if the user provides its custom pulse gate ansatz).

To improve computation times, we build here some small evolutions of two main classes enabling the pulse level
simulation
of our workflow: ```BackendEstimator``` and ```Solver```.

Before introducing the new replacing classes, let's recap here how the pulse level simulation workflow is handled with
the baseline workflow:

- ```TorchQuantumEnvironment```: This is the main class modelling the entire RL environment, controlling what is going
  to be applied on it through
  actions sent by the agent to parametrize the execution. In this class, we handle everything that enables us to compute
  a reward from the measurement counts enabling us to perform a successful gate calibration.
  There are three main steps enabling such gate calibration for a given circuit context. At each iteration of the
  training:
    1. the environment is randomly set into a different initial state, sampled from the
       Pauli preparation basis in order to have a tomographically complete set of states for ensuring a successful gate
       calibration.
    2. the environment determines a set of Pauli observables to be measured experimentally that enable the estimation of
       the state fidelity with respect to the target state of the quantum circuit (deduced from the initial state
       knowledge and the ideal unitary we would like to compute).
    3. it calls the pre-defined ``Estimator`` primitive in charge of computing the expectation values of provided
       observables in the state yielded by the parametrized input quantum circuits, filled with parameter values coming
       from the agent.


- ```BackendEstimator```: This ``Estimator`` subclass is a simple wrapper of a call to the method ```backend.run()```,
  enabling the
  extraction
  of observables expectation values through pre- and post-processing of the input circuits. Typically, pre-processing
  consists
  in building adapted versions of the input ```circuits```(with potential ```parameter_values```) enabling us to measure
  in the appropriate Pauli basis to extract from
  the counts the value of the desired input ``observables``. The main part of the code is to call
  the ```backend.run()```
  method over
  the list of those adapted circuits. The post-processing retrieves the counts information
  associated to the circuits execution and converts those in expectation values.


- ```DynamicsBackend```:In the case of our pulse level simulation, ``backend`` is typically a ```DynamicsBackend```
  object,
  and we can check that this backend, once baseline gates such as S and H are calibrated, just converts the
  input ```List[QuantumCircuit]```object to a
  list of corresponding pulse ```List[Schedule]```, and then calls the method ```Solver.solve``` over this list of
  schedules. One important thing to note here is that on top of calling the ```Solver```, the ```DynamicsBackend```also
  carries
  the choice of initial quantum state from which the simulations should start. In the general case for quantum circuits,
  we always start
  in the ground state. However, acknowledging that we can modify this initial state at will through one of the
  options of the backend will be helpful to enhance simulations performance as we have a random different starting point
  at the beginning of each iteration.

The ```Solver```, when working with Jax, will try to build a jit function taking the samples issues from each individual
pulse schedule and simulate each schedule independently. The main issue arising here, is that although simulation is
accelerated by being done inside Jax,
the task of converting down schedules to samples (```InstructiontoSignal``` converter) is done outside the jit compiled
function
to keep things generic (because Qiskit Pulse is not fully JAX compatible yet). This process is extremely costly when the
circuit starts to be a few gates long, as the
arrays involved are quite big.

To address this issue, we would like to enable the user to provide its own custom jit
function that will be called internally by the ```Solver``` in order to maintain the workflow as similar as possible to
the usual
usage of the ```TorchQuantumEnvironment``` class.

To proceed, we use the fact that we have in general a few numbers of circuit truncations only to test out, and that we
can therefore
build each truncation individually at the pulse level through user-defined functions implementing the circuit they want
through the method
```pulse.call()```, which enables to build modular pulse schedules in a similar fashion the gate model.

## The new classes

Our will is to minimize the required changes of the interface so that the user can maintain most of the workflow
identical.
However, there are two specific classes that will need to be changed to leverage the optimized Jax execution.

1. ``DynamicsBackendEstimator``: Inheriting from ``BackendEstimator`, this class is strictly equivalent from the point
   of view of the user.
   In fact, the user does not have to care about this change, as it is done within the Environment declaration.
   Although not relevant for the user interface, we mention here what changes compared to the base class.
   Moreover, this new classe also adds some ```solver_options``` (one of the ```DynamicsBackend```options, passed to
   the ```Solver.solve``` method) enabling the simulation such as:
    1. Information about the observables to be measured through the measurement circuits to be run for each iteration.
    2. The parameter values to bind to the parametrized circuit/schedule
2. ```JaxSolver```: This is the main change for the user, as this custom ``Solver`` takes, on top of the existing
   arguments required to initialize the baseline ``Solver``, a function or list of functions as argument, which
   corresponds to the pulse
   translation of the target circuit(s) one wants to run.
3. We show in one of the notebooks how those functions should be defined for a pre-defined target circuit.

