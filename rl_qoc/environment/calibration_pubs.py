from __future__ import annotations

from qiskit.circuit import CircuitInstruction
from qiskit.primitives import PrimitiveResult
from qiskit.primitives.containers.estimator_pub import EstimatorPub, EstimatorPubLike
from qiskit.primitives.containers.sampler_pub import SamplerPub, SamplerPubLike
from numbers import Real, Integral
from collections.abc import Mapping
from typing import Tuple, Union, Iterable, List

from qiskit import QuantumCircuit

from qiskit.primitives.containers.bindings_array import BindingsArray, BindingsArrayLike
from qiskit.primitives.containers.observables_array import ObservablesArray, ObservablesArrayLike

class CalibrationEstimatorPub(EstimatorPub):
    """
    Subclass of Qiskit EstimatorPub class to store additional metadata that could be leveraged
    for optimal unrolling in control system.
    """
    __slots__ = ("_circuit", "_observables", "_parameter_values", "_precision", "_shape", "_n_reps", "_input_states")
    
    def __init__(self,
                 circuit: QuantumCircuit,
                 observables: ObservablesArray,
                 parameter_values: BindingsArray | None = None,
                 precision: float | None = None,
                 n_reps: int = 1,
                 input_states: Iterable[QuantumCircuit] | QuantumCircuit| None = None,
                 validate: bool = True):
        """
        Args:
            circuit: A quantum circuit.
            observables: An observables array.
            parameter_values: A bindings array, if the circuit is parametric.
            precision: An optional target precision for expectation value estimates.
            n_reps: Number of repetitions of the circuit.
            input_states: Input states of the circuit (expressed as a List of QuantumCircuits).
            validate: Whether to validate arguments during initialization.
        """

        self._n_reps = int(n_reps)
        if input_states is None:
            self._input_states = [circuit.copy_empty_like()]
        elif isinstance(input_states, QuantumCircuit):
            self._input_states = [input_states]
        elif all(isinstance(input_state, QuantumCircuit) for input_state in input_states):
            self._input_states = list(input_states)
        super().__init__(circuit, observables, parameter_values, precision, validate)
        
    
    def to_pub_list(self) -> List[EstimatorPub]:
        """
        Convert the CalibrationEstimatorPub object to a list of EstimatorPub objects.
        """
        pub_list = []
        for input_state in self.input_states:
            circuit = self._circuit.repeat(self.n_reps)
            circuit.compose(input_state, front=True, inplace=True)
            pub_list.append(EstimatorPub(circuit,
                                         self._observables,
                                         self._parameter_values,
                                         self._precision))
        
        return pub_list
    
    def validate(self):
        """
        Validate the CalibrationEstimatorPub object.
        """
        super().validate()
        if not isinstance(self._n_reps, int):
            raise TypeError(f"n_reps must be an integer, not {type(self._n_reps)}")
        if not isinstance(self._input_states, list):
            raise TypeError(f"input_states must be a list, not {type(self._input_states)}")
        if not all(isinstance(input_state, QuantumCircuit) for input_state in self._input_states):
            raise TypeError("All elements of input_states must be QuantumCircuits.")
        
    
    @classmethod
    def coerce(cls, pub: CalibrationEstimatorPubLike, precision: float | None = None) -> CalibrationEstimatorPub:
        """
        Coerce a CalibrationEstimatorPubLike object to a CalibrationEstimatorPub object.
        
        Args:
            pub: A compatible object for coercion.
            precision: An optional default precision to use if not already specified by the pub-like object.
            
        Returns:
            An estimator pub.
        """
        # Validate precision kwarg if provided
        if precision is not None:
            if not isinstance(precision, Real):
                raise TypeError(f"precision must be a real number, not {type(precision)}.")
            if precision < 0:
                raise ValueError("precision must be non-negative")
        if isinstance(pub, CalibrationEstimatorPub):
            if pub.precision is None and precision is not None:
                return cls(pub.circuit,
                            pub.observables,
                            pub.parameter_values,
                            precision,
                            pub.n_reps,
                            pub.input_states,
                            validate=False)
            return pub
            
        elif isinstance(pub, EstimatorPub):
            if pub.precision is None and precision is not None:
                return cls(pub.circuit,
                           pub.observables,
                           pub.parameter_values,
                            precision,
                           validate=False)
        if isinstance(pub, QuantumCircuit):
            raise ValueError(
                f"An invalid Estimator pub-like was given ({type(pub)}). "
                "If you want to run a single pub, you need to wrap it with `[]` like "
                "`estimator.run([(circuit, observables, param_values)])` "
                "instead of `estimator.run((circuit, observables, param_values))`."
            )
        if len(pub) not in [2, 3, 4, 5, 6]:
            raise ValueError(
                f"The length of pub must be 2, 3, 4, 5 or 6, but length {len(pub)} is given."
            )
        circuit = pub[0]
        observables = ObservablesArray.coerce(pub[1])
        
        if len(pub) > 2 and pub[2] is not None:
            values = pub[2]
            if not isinstance(values, (BindingsArray, Mapping)):
                values = {tuple(circuit.parameters): values}
            parameter_values = BindingsArray.coerce(values)
        else:
            parameter_values = None
            
        if len(pub) > 3 and pub[3] is not None:
            precision = pub[3]
            
        if len(pub) > 4 and pub[4] is not None:
            n_reps = pub[4]
        else:
            n_reps = 1
            
        if len(pub) > 5 and pub[5] is not None:
            input_states = pub[5]
        else:
            input_states = [circuit.copy_empty_like()]
            
        return cls(circuit,
                     observables,
                     parameter_values,
                     precision,
                     n_reps,
                     input_states,
                     validate=True)
        
            
    @property
    def n_reps(self) -> int:
        """
        Number of repetitions of the circuit.
        """
        return self._n_reps
    
    @property
    def input_states(self) -> List[QuantumCircuit]:
        """
        Input states of the circuit.
        """
        return self._input_states
    

class CalibrationSamplerPub(SamplerPub):
    """
    Subclass of Qiskit SamplerPub class to store additional metadata that could be leveraged
    for optimal unrolling in control system.
    """
    __slots__ = ("_circuit", "_parameter_values", "_shots", "_shape", "_n_reps", "_input_states")
    
    def __init__(self,
                 circuit: QuantumCircuit,
                 parameter_values: BindingsArray | None = None,
                 shots: int | None = None,
                 n_reps: int = 1,
                 input_states: Iterable[QuantumCircuit] | QuantumCircuit | None = None,
                 validate: bool = True):
        """
        Args:
            circuit: A quantum circuit.
            parameter_values: A bindings array.
            shots: A specific number of shots to run with. This value takes
                precedence over any value owed by or supplied to a sampler.
            n_reps: Number of repetitions of the circuit.
            input_states: Input states of the circuit (expressed as a List of QuantumCircuits for each n_reps).
            validate: If ``True``, the input data is validated during initialization.
        """
        self._n_reps = int(n_reps)
        if input_states is None:
            self._input_states = [circuit.copy_empty_like()]
        elif isinstance(input_states, QuantumCircuit):
            self._input_states = [input_states]
        elif all(isinstance(input_state, QuantumCircuit) for input_state in input_states):
            self._input_states = list(input_states)
        super().__init__(circuit, parameter_values, shots, validate)  
        
    
    def to_pub_list(self) -> List[SamplerPub]:
        """
        Convert the CalibrationSamplerPub object to a list of SamplerPub objects.
        """
        pub_list = []
        for input_state in self.input_states:
            circuit = self._circuit.repeat(self.n_reps)
            if input_state.data:
                circuit.compose(input_state, front=True, inplace=True)
            pub_list.append(SamplerPub(circuit,
                                       self._parameter_values,
                                       self._shots))
        return pub_list

    def validate(self):
        """
        Validate the CalibrationEstimatorPub object.
        """
        super().validate()
        if not isinstance(self._n_reps, int):
            raise TypeError(f"n_reps must be an integer, not {type(self._n_reps)}")
        if not isinstance(self._input_states, list):
            raise TypeError(f"input_states must be a list, not {type(self._input_states)}")
        if not all(isinstance(input_state, QuantumCircuit) for input_state in self._input_states):
            raise TypeError("All elements of input_states must be QuantumCircuits.")
        

    @classmethod
    def coerce(cls, pub: CalibrationSamplerPubLike, shots: int | None = None) -> CalibrationSamplerPub:
        """
        Coerce a CalibrationSamplerPubLike object to a CalibrationSamplerPub object.

        Args:
            pub: A compatible object for coercion.
            shots: An optional default number of shots to use if not
                   already specified by the pub-like object.

        Returns:
            An Sampler pub.
        """
        # Validate precision kwarg if provided
        if shots is not None:
            if not isinstance(shots, Integral) or isinstance(shots, bool):
                raise TypeError("shots must be an integer")
            if shots <= 0:
                raise ValueError("shots must be positive")
        if isinstance(pub, CalibrationSamplerPub):
            if pub.shots is None and shots is not None:
                return cls(pub.circuit,
                           pub.parameter_values,
                           shots,
                           pub.n_reps,
                           pub.input_states,
                           validate=False)
            return pub

        elif isinstance(pub, EstimatorPub):
            if pub.precision is None and shots is not None:
                return cls(pub.circuit,
                           pub.parameter_values,
                           shots,
                           validate=False)
        if isinstance(pub, CircuitInstruction):
            raise ValueError(
                f"An invalid Sampler pub-like was given ({type(pub)}). "
                "If you want to run a single circuit, "
                "you need to wrap it with `[]` like `sampler.run([circuit])` "
                "instead of `sampler.run(circuit)`."
            )
        if len(pub) not in [2, 3, 4, 5, 6]:
            raise ValueError(
                f"The length of pub must be 2, 3, 4, 5 or 6, but length {len(pub)} is given."
            )
        circuit = pub[0]

        if len(pub) > 1 and pub[1] is not None:
            values = pub[1]
            if not isinstance(values, (BindingsArray, Mapping)):
                values = {tuple(circuit.parameters): values}
            parameter_values = BindingsArray.coerce(values)
        else:
            parameter_values = None

        if len(pub) > 2 and pub[2] is not None:
            shots = pub[2]

        if len(pub) > 3 and pub[3] is not None:
            n_reps = pub[3]
        else:
            n_reps = 1

        if len(pub) > 4 and pub[4] is not None:
            input_states = pub[4]
        else:
            input_states = [circuit.copy_empty_like()]

        return cls(circuit,
                   parameter_values,
                   shots,
                   n_reps,
                   input_states,
                   validate=True)

    @property
    def n_reps(self) -> int:
        """
        Number of repetitions of the circuit.
        """
        return self._n_reps

    @property
    def input_states(self) -> List[QuantumCircuit]:
        """
        Input states of the circuit.
        """
        return self._input_states
    
        
CalibrationEstimatorPubLike = Union[EstimatorPubLike,
                                    Tuple[QuantumCircuit, ObservablesArrayLike, BindingsArrayLike, Real,
                                    Union[int, Iterable[int]], Union[Iterable[QuantumCircuit], None]]]         

CalibrationSamplerPubLike = Union[SamplerPubLike, 
                                    Tuple[QuantumCircuit, BindingsArrayLike, Union[Integral, None],
                                            Union[int, Iterable[int]], Union[Iterable[QuantumCircuit], None]]]
    
