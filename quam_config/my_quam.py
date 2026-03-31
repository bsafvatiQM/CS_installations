from dataclasses import field
from pathlib import Path
from typing import ClassVar, Dict, Optional, Type

from quam.core import quam_dataclass
from quam.serialisation import JSONSerialiser
from quam.components.channels import SingleChannel
from quam.components.hardware import BaseFrequencyConverter
from quam_builder.architecture.superconducting.qpu import FixedFrequencyQuam
from quam_builder.architecture.superconducting.qubit import FixedFrequencyTransmon
from quam_builder.architecture.superconducting.qubit_pair import (
    FixedFrequencyTransmonPair,
)
from quam_builder.architecture.superconducting.components.tunable_coupler import (
    TunableCoupler,
)

QUAM_STATE_DIR = Path(__file__).resolve().parent.parent / "quam_state"


@quam_dataclass
class SSBModulatorChannel(SingleChannel):
    """Single-sideband modulator channel: one OPX output + external LO.

    Mirrors IQChannel's frequency management (LO/RF/IF triplet) but generates
    a singleInput QUA element instead of mixInputs.  The OPX produces the IF
    tone and the external SSB modulator upconverts it using its own LO.

    Args:
        frequency_converter_up: FrequencyConverter holding the LocalOscillator.
        LO_frequency: Local oscillator frequency (Hz). Defaults to the value
            stored on the frequency converter.
        RF_frequency: Target RF frequency (Hz). Defaults to LO + IF.
    """

    frequency_converter_up: Optional[BaseFrequencyConverter] = None

    LO_frequency: float = "#./frequency_converter_up/LO_frequency"
    RF_frequency: Optional[float] = None
    intermediate_frequency: float = "#./inferred_intermediate_frequency"

    @property
    def inferred_intermediate_frequency(self) -> float:
        if self.RF_frequency is not None and self.LO_frequency is not None:
            return self.RF_frequency - self.LO_frequency
        return 0.0

    @property
    def inferred_RF_frequency(self) -> float:
        if self.LO_frequency is not None:
            return self.LO_frequency + self.intermediate_frequency
        return None


@quam_dataclass
class FixedTransmonPairWithCoupler(FixedFrequencyTransmonPair):
    """FixedFrequencyTransmonPair extended with an optional tunable coupler.

    The upstream class only declares coupler on FluxTunableTransmonPair.
    This local subclass adds it so fixed-frequency pairs with coupler flux
    lines serialize and deserialize correctly.
    """

    coupler: Optional[TunableCoupler] = None


@quam_dataclass
class Quam(FixedFrequencyQuam):
    qubit_pair_type: ClassVar[Type] = FixedTransmonPairWithCoupler
    qubit_pairs: Dict[str, FixedTransmonPairWithCoupler] = field(
        default_factory=dict
    )

    @classmethod
    def get_serialiser(cls) -> JSONSerialiser:
        return JSONSerialiser(
            content_mapping={"wiring": "wiring.json", "network": "wiring.json"},
            state_path=QUAM_STATE_DIR,
        )
