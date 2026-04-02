from dataclasses import field
from pathlib import Path
from typing import Dict

from quam.core import quam_dataclass
from quam.components.channels import SingleChannel
from quam.serialisation import JSONSerialiser
from quam_builder.architecture.superconducting.qubit import FluxTunableTransmon
from quam_builder.architecture.superconducting.qpu import FluxTunableQuam

QUAM_STATE_DIR = Path(__file__).resolve().parent.parent / "quam_state"


@quam_dataclass
class DualZDriveTransmon(FluxTunableTransmon):
    z_ac: SingleChannel = None


@quam_dataclass
class Quam(FluxTunableQuam):
    qubits: Dict[str, DualZDriveTransmon] = field(default_factory=dict)

    @classmethod
    def get_serialiser(cls) -> JSONSerialiser:
        return JSONSerialiser(
            content_mapping={"wiring": "wiring.json", "network": "wiring.json"},
            state_path=QUAM_STATE_DIR,
        )
