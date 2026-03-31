"""
QUA macros for the Murch blue fridge system.

These are called inside a QUA `with program() as prog:` block.
They emit QUA statements — they do not return values.

Y gates are implemented via frame rotation of the corresponding X gates:
    Y(θ) = Rz(π/2) · X(θ) · Rz(-π/2)
"""

from qm.qua import frame_rotation_2pi


def y180(qubit):
    """Y180 gate: frame-rotate +π/2, play x180, frame-rotate -π/2."""
    frame_rotation_2pi(0.25, qubit.xy.name)
    qubit.xy.play("x180")
    frame_rotation_2pi(-0.25, qubit.xy.name)


def y90(qubit):
    """Y90 gate: frame-rotate +π/2, play x90, frame-rotate -π/2."""
    frame_rotation_2pi(0.25, qubit.xy.name)
    qubit.xy.play("x90")
    frame_rotation_2pi(-0.25, qubit.xy.name)


def minus_y90(qubit):
    """-Y90 gate: frame-rotate -π/2, play x90, frame-rotate +π/2."""
    frame_rotation_2pi(-0.25, qubit.xy.name)
    qubit.xy.play("x90")
    frame_rotation_2pi(0.25, qubit.xy.name)
