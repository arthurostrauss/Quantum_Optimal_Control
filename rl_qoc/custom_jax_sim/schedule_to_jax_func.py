import jax
import jax.numpy as jnp
import sympy
from qiskit.circuit import ParameterExpression
from qiskit.pulse import Schedule, ScheduleBlock, Play, ShiftPhase, SetPhase, Delay, SetFrequency, ShiftFrequency
from qiskit.pulse.library import SymbolicPulse, ScalableSymbolicPulse, Waveform

def build_jax_schedule_reconstructor(template_schedule):
    """
    Parses a Qiskit Schedule and returns a JAX-compatible reconstruction function.
    
    Fixes:
    1. Robust casting of parameter dependencies to sets (fixes AttributeError).
    2. Explicit support for multi-parameter expressions (e.g., amp = p1 + p2).
    """
    
    # 1. Parameter Extraction & Indexing
    target_parameters = sorted(template_schedule.parameters, key=lambda p: p.name)
    target_param_map = {p: i for i, p in enumerate(target_parameters)}
    target_param_names = [p.name for p in target_parameters]
    target_param_set = set(target_parameters)

    # 2. Helper: Transpiler for ParameterExpressions
    def analyze_parameter_value(val):
        """
        Analyzes a value (float or ParameterExpression).
        
        Returns:
            is_dynamic (bool): True if it depends on target_parameters.
            evaluator (callable | None): A JAX-ready function f(args) -> result.
            dependency_indices (list): Indices of input array required by evaluator.
            static_value: The value if it is static, else None.
        """
        # Static Check
        if not isinstance(val, ParameterExpression):
            return False, None, [], val

        # --- FIX STARTS HERE ---
        # Explicitly cast to set to handle dict_keys, lists, or other iterables
        val_params = set(val.parameters)
        dependencies = val_params.intersection(target_param_set)
        # --- FIX ENDS HERE ---
        
        if not dependencies:
            # Expression exists but uses parameters we aren't optimizing (constants)
            try:
                return False, None, [], float(val)
            except TypeError:
                return False, None, [], complex(val)

        # Multi-Parameter Support Logic:
        # 1. Sort dependencies. If val = p1 + p2, we get [p1, p2]
        sorted_deps = sorted(list(dependencies), key=lambda p: p.name)
        
        # 2. Map these specific dependencies to indices in the MAIN input array
        dependency_indices = [target_param_map[p] for p in sorted_deps]
        
        # 3. Create JAX function. 
        # If val depends on multiple parameters, sym_args has length > 1.
        sym_expr = sympy.sympify(str(val))
        sym_args = [sympy.Symbol(p.name) for p in sorted_deps]
        
        # lambdify creates a function f(arg1, arg2, ...) -> result
        jax_evaluator = sympy.lambdify(sym_args, sym_expr, modules=jnp)
        
        return True, jax_evaluator, dependency_indices, None

    # 3. Instruction Blueprinting
    instruction_blueprints = []
    
    # Normalize iteration (handling both Schedule and ScheduleBlock)
    if isinstance(template_schedule, ScheduleBlock):
        iterable_insts = template_schedule.instructions
    else:
        # Schedule.instructions yields (time, inst), ignore time for reconstruction
        iterable_insts = [inst for _, inst in template_schedule.instructions]
    
    for inst in iterable_insts:
        if isinstance(inst, Play):
            if isinstance(inst.pulse, Waveform):
                raise ValueError("Fixed 'Waveform' objects cannot be JAX accelerated.")
            
            pulse_instance = inst.pulse
            static_params = {}
            dynamic_handlers = {} 
            
            for p_name, p_val in pulse_instance.parameters.items():
                is_dyn, evaluator, deps, static_val = analyze_parameter_value(p_val)
                
                if is_dyn:
                    dynamic_handlers[p_name] = (evaluator, deps)
                else:
                    static_params[p_name] = static_val
            
            instruction_blueprints.append({
                'type': 'play_symbolic',
                'channel': inst.channel,
                'envelope': pulse_instance.envelope,
                'pulse_type': pulse_instance.pulse_type,
                'duration': pulse_instance.duration, 
                'static_params': static_params,
                'dynamic_handlers': dynamic_handlers
            })

        elif isinstance(inst, (ShiftPhase, SetPhase)):
            is_dyn, evaluator, deps, static_val = analyze_parameter_value(inst.phase)
            instruction_blueprints.append({
                'type': 'phase',
                'class': type(inst), 
                'channel': inst.channel,
                'is_dynamic': is_dyn,
                'evaluator': evaluator,
                'deps': deps,
                'static_val': static_val
            })
        elif isinstance(inst, (SetFrequency, ShiftFrequency)):
            is_dyn, evaluator, deps, static_val = analyze_parameter_value(inst.frequency)
            instruction_blueprints.append({
                'type': 'frequency',
                'class': type(inst),
                'channel': inst.channel,
                'is_dynamic': is_dyn,
                'evaluator': evaluator,
                'deps': deps,
                'static_val': static_val
            })
        elif isinstance(inst, Delay):
            is_dyn, _, _, _ = analyze_parameter_value(inst.duration)
            if is_dyn:
                 raise ValueError("JAX JIT requires static durations for Delays.")
            instruction_blueprints.append({'type': 'static', 'inst': inst})
        else:
            instruction_blueprints.append({'type': 'static', 'inst': inst})

    # 4. The JAX Reconstructor
    def reconstruct_schedule(param_values):
        """
        Args:
            param_values (jax.numpy.array): Input array of parameter values.
        """
        # Use ScheduleBlock for mutability and standard formatting
        sched = ScheduleBlock(name="jax_generated", 
                              alignment_context=template_schedule.alignment_context 
                              if isinstance(template_schedule, ScheduleBlock) else None)
        
        for bp in instruction_blueprints:
            if bp['type'] == 'static':
                sched.append(bp['inst'], inplace=True)
                
            elif bp['type'] == 'play_symbolic':
                current_params = bp['static_params'].copy()
                
                # Evaluate Dynamic Parameters
                for p_name, (evaluator, deps) in bp['dynamic_handlers'].items():
                    # Support for Multi-Parameter connections:
                    # We extract ALL required args from the main array
                    args = [param_values[i] for i in deps]
                    
                    # Pass *args to the evaluator (handles f(x), f(x,y), etc.)
                    val = evaluator(*args)
                    current_params[p_name] = val
                
                # Instantiate Pulse
                new_pulse = ScalableSymbolicPulse(
                    pulse_type=bp['pulse_type'],
                    duration=bp['duration'],
                    amp=current_params.get('amp', 1.0),
                    angle=current_params.get('angle', 0.0),
                    parameters=current_params,   
                    envelope=bp['envelope'],     
                    limit_amplitude=False        
                )
                sched.append(Play(new_pulse, bp['channel']), inplace=True)
            
            elif bp['type'] == 'phase':
                if bp['is_dynamic']:
                    args = [param_values[i] for i in bp['deps']]
                    phase_val = bp['evaluator'](*args)
                else:
                    phase_val = bp['static_val']
                sched.append(bp['class'](phase_val, bp['channel']), inplace=True)
                
        return sched

    return reconstruct_schedule, target_param_names