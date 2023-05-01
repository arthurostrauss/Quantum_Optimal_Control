# Feedback-based Quantum Control using Reinforcement Learning 

This repository is dedicated to the writing of Python scripts enabling the realization of quantum control tasks based on
closed-loop optimization that incorporates measurements. We build upon the work of Vlad Sivak on Model-Free Quantum
Control with Reinforcement Learning (RL) to enable a generic framework for arbitrary state preparation and quantum gate 
calibration based on Qiskit modules.

The repo is currently divided in three main folders. The first one is the folder "paper_results", where adaptation of 
the state preparation algorithm developed by Sivak for bosonic systems is adapted to a multiqubit system simulated by a
Qiskit ``` QuantumCircuit``` running on a ```QasmSimulator``` backend instance.

We briefly explain the overall logic of the algorithm below.

## Policy Gradient algorithm

Reinforcement Learning is an interaction-based learning procedure resulting from the interaction between two entities:
- an Environment: in our case the quantum system of interest, characterized by a quantum state and from which 
observations/measurements can be extracted.
- an Agent: a classical Neural Network on the PC, whose goal is to learn a policy (probability distribution) from which
actions are sampled to be applied on the Environment in order to set it in a specific target state. Typically, the goal 
of the Agent in the frame of Quantum Optimal Control is to find actions (e.g. circuit or pulse parameters)
that will enable the successful preparation of a target quantum state, or the calibration of a quantum operation.


