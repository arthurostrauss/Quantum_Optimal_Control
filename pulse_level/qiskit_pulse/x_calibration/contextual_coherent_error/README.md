# Contextual coherent error suppression

The model of the coherent error is widely known at the circuit level, as it consists in
specifying a unitary matrix that represents the error. However, the error can be contextual,
i.e., it can depend on the gate type/sequence/parameter. In this case, the error is not a
fixed unitary matrix, but a function of the gate type/sequence/parameter. This is the case of
this error model, which we try to suppress through our RL agent.

We focus on the following gate: a single-qubit $RX(\theta)$ rotation. The error is a function of the
angle of the rotation, and it is represented by a unitary matrix $U(\gamma,\theta) = RX(\gamma \theta)$ where $\gamma$
is the strength factor. We model this error at the pulse level, assuming that we can create a $RX$ gate by applying a
Rabi
driving pulse with an amplitude proportional to the angle of the rotation. The error is then a function of the amplitude
of the pulse.
