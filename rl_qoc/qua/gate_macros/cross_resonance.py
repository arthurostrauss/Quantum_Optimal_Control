from typing import Optional, Literal
from qm.qua import *
from quam.core import quam_dataclass
from quam.components.macro import QubitPairMacro
from qm.qua._dsl import QuaExpression, QuaVariable

__all__ = ["CRGate"]


qua_T = QuaVariable | QuaExpression


@quam_dataclass
class CRGate(QubitPairMacro):
    qc_correction_phase: float = 0.0

    def apply(
            self,
            cr_type: Literal[
                "direct", "direct+cancel", "direct+echo", "direct+cancel+echo"
            ] = "direct",
            wf_type: Literal["square", "cosine", "gauss", "flattop"] = "square",
            cr_drive_amp_scaling: Optional[float | qua_T] = None,
            cr_drive_phase: Optional[float | qua_T] = None,
            cr_cancel_amp_scaling: Optional[float | qua_T] = None,
            cr_cancel_phase: Optional[float | qua_T] = None,
            cr_duration_clock_cycles: Optional[float | qua_T] = None,
            qc_correction_phase: Optional[float | qua_T] = None,
    ) -> None:
        qc = self.qubit_pair.qubit_control
        qt = self.qubit_pair.qubit_target
        cr = self.qubit_pair.cross_resonance
        cr_elems = [qc.xy.name, qt.xy.name, cr.name]

        def _play_cr_pulse(
                elem,
                wf_type: str = wf_type,
                amp_scale: Optional[float | qua_T] = None,
                duration: Optional[float | qua_T] = None,
                sgn: int = 1,
        ):
            if amp_scale is None and duration is None:
                elem.play(wf_type)
            elif amp_scale is None:
                elem.play(wf_type, duration=duration)
            elif duration is None:
                elem.play(wf_type, amplitude_scale=sgn * amp_scale)
            else:
                elem.play(wf_type, amplitude_scale=sgn * amp_scale, duration=duration)

        def cr_drive_shift_phase():
            if cr_drive_phase is not None:
                cr.frame_rotation_2pi(cr_drive_phase)

        def cr_cancel_shift_phase():
            if cr_cancel_phase is not None:
                qt.xy.frame_rotation_2pi(cr_cancel_phase)

        def qc_shift_correction_phase():
            if qc_correction_phase is not None:
                qc.xy.frame_rotation_2pi(qc_correction_phase)

        def cr_drive_play(
                sgn: Literal["direct", "echo"] = "direct",
                wf_type=wf_type,
        ):
            _play_cr_pulse(
                elem=cr,
                wf_type=wf_type,
                amp_scale=cr_drive_amp_scaling,
                duration=cr_duration_clock_cycles,
                sgn=1 if sgn == "direct" else -1,
            )

        def cr_cancel_play(
                sgn: Literal["direct", "echo"] = "direct",
                wf_type=wf_type,
        ):
            _play_cr_pulse(
                elem=qt.xy,
                wf_type=f"cr_{wf_type}_{self.qubit_pair.name}",
                amp_scale=cr_cancel_amp_scaling,
                duration=cr_duration_clock_cycles,
                sgn=1 if sgn == "direct" else -1,
            )

        if cr_type == "direct":
            cr_drive_shift_phase()
            align(*cr_elems)

            cr_drive_play(sgn="direct")
            align(*cr_elems)

            reset_frame(cr.name)
            qc_shift_correction_phase()
            align(*cr_elems)

        elif cr_type == "direct+echo":
            cr_drive_shift_phase()
            align(*cr_elems)

            cr_drive_play(sgn="direct")
            align(*cr_elems)

            qc.xy.play("x180")
            align(*cr_elems)

            cr_drive_play(sgn="echo")
            align(*cr_elems)

            qc.xy.play("x180")
            align(*cr_elems)

            reset_frame(cr.name)
            qc_shift_correction_phase()
            align(*cr_elems)

        elif cr_type == "direct+cancel":
            cr_drive_shift_phase()
            cr_cancel_shift_phase()
            align(*cr_elems)

            cr_drive_play(sgn="direct")
            cr_cancel_play(sgn="direct")
            align(*cr_elems)

            reset_frame(cr.name)
            reset_frame(qt.xy.name)
            qc_shift_correction_phase()
            align(*cr_elems)

        elif cr_type == "direct+cancel+echo":
            cr_drive_shift_phase()
            cr_cancel_shift_phase()
            align(*cr_elems)

            cr_drive_play(sgn="direct")
            cr_cancel_play(sgn="direct")
            align(*cr_elems)

            qc.xy.play("x180")
            align(*cr_elems)

            cr_drive_play(sgn="echo")
            cr_cancel_play(sgn="echo")
            align(*cr_elems)

            qc.xy.play("x180")
            align(*cr_elems)

            reset_frame(cr.name)
            reset_frame(qt.xy.name)
            qc_shift_correction_phase()
            align(*cr_elems)