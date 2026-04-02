"""
Populate the QUAM with calibrated values loaded from a JSON file.

This script mirrors the parameter structure of sample_config_black.py and loads
every attribute it reads from the calibration JSON (e.g. latest_values_xiayu11.json).

Parameters that map directly to the QUAM data model are applied in-place.
Parameters that have no QUAM equivalent (ef transitions, parametric drives, ACS,
FFL, extra mixer corrections) are collected into `calibrated_extras.json` for use
by downstream experiment scripts.

System (physical wiring — identical to populate_murch.py / sample_config_black.py):
  - 3 fixed-frequency transmons (Q1, Q2, Q3/QC) in a linear chain
  - 2 tunable couplers (C12 on port 1, C23 on port 2) with flux lines
  - Q1 drive: IQ mixer on ports 3/4  (lo_qubit = 4.241 GHz)
  - Q2 drive: SSB modulator on ports 5/6  (lo_qubit = 4.241 GHz)
  - Q3 drive: SSB modulator on ports 9/10, shared with readout (lo_mod = 5.347 GHz)
  - Multiplexed readout: IQ mixer on ports 9/10, ADC on inputs 1/2 (lo_res = 6.982 GHz)

Qubit name mapping (QUAM ↔ sample_config_black):
  q1 ↔ Q1 / Q1_DL      q2 ↔ Q2 / Q2_DL      q3 ↔ QC
"""

########################################################################################################################
# %%                                             Import section
########################################################################################################################
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from qualang_tools.units import unit
from quam_config import Quam
from quam_builder.builder.superconducting.pulses import add_DragCosine_pulses
from quam.components.channels import DigitalOutputChannel, SingleChannel

########################################################################################################################
# %%                                          Load calibrated values
########################################################################################################################
JSON_FILE = Path(r"C:/Users/BenjaminSafvati/Downloads/shared_folder/shared_folder/latest_values_xiayu11.json")

with open(JSON_FILE, "r") as f:
    lv = json.load(f)  # shorthand used throughout

########################################################################################################################
# %%                                 QUAM loading and auxiliary functions
########################################################################################################################
machine = Quam.load()
u = unit(coerce_to_integer=True)


def correction_matrix_to_gain_phase(matrix):
    """Invert the IQ_imbalance formula to recover (gain, phase) from a 4-element
    correction matrix [c00, c01, c10, c11].

    IQ_imbalance(g, phi) produces:
        c00 = N*(1-g)*cos(phi),  c01 = N*(1+g)*sin(phi)
        c10 = N*(1-g)*sin(phi),  c11 = N*(1+g)*cos(phi)
    where N = 1 / ((1-g^2)*(2*cos^2(phi)-1)).

    Inverting:
        g   = (c11 - c00) / (c11 + c00)
        phi = arctan2(c10, c00)
    """
    c00, c01, c10, c11 = matrix
    denom = c11 + c00
    gain = float((c11 - c00) / denom) if abs(denom) > 1e-15 else 0.0
    phase = float(np.arctan2(c10, c00)) if abs(c00) > 1e-15 else 0.0
    return gain, phase


########################################################################################################################
# %%                         Fixed LO frequencies (hardcoded in sample_config_black.py)
########################################################################################################################
lo_res = 6_982_000_000
lo_qubit = 4_241_000_000
lo_mod = 5_347_000_000

########################################################################################################################
# %%  1. Resonator frequencies & readout parameters
########################################################################################################################
res_freq = [lv["res_1_freq"], lv["res_C_freq"], lv["res_2_freq"]]
readout_amp = [lv["amp_readout_res_1"], lv["amp_readout_res_C"], lv["amp_readout_res_2"]]
readout_len = [lv["time_readout_res_1"], lv["time_readout_res_C"], lv["time_readout_res_2"]]

for k, qubit in enumerate(machine.qubits.values()):
    qubit.resonator.f_01 = float(res_freq[k])
    qubit.resonator.RF_frequency = float(res_freq[k])
    qubit.resonator.frequency_converter_up.LO_frequency = lo_res
    qubit.resonator.operations["readout"].amplitude = float(readout_amp[k])
    qubit.resonator.operations["readout"].length = int(readout_len[k])

########################################################################################################################
# %%  2. Qubit drive frequencies
########################################################################################################################
qubit_freq = [lv["Q1_freq"], lv["QC_freq"], lv["Q2_freq"]]
qubit_ef_freq = [lv["Q1_ef_freq"], lv["QC_ef_freq"], lv["Q2_ef_freq"]]
qubit_lo = [lo_qubit, lo_mod, lo_qubit]
anharmonicity = [float(qubit_ef_freq[k] - qubit_freq[k]) for k in range(3)]

for k, qubit in enumerate(machine.qubits.values()):
    qubit.f_01 = float(qubit_freq[k])
    qubit.xy.RF_frequency = float(qubit_freq[k])
    qubit.xy.frequency_converter_up.LO_frequency = float(qubit_lo[k])
    qubit.grid_location = f"{k},0"

########################################################################################################################
# %%  3. DC offsets on analog output ports
########################################################################################################################
dc_offset_map = {
    1: lv["DC_offset1"],
    2: lv["DC_offset2"],
    3: lv["DC_offset3"],
    4: lv["DC_offset4"],
    5: lv["DC_offset5"],
    6: lv["DC_offset6"],
    9: lv["DC_offset9"],
    10: lv["DC_offset10"],
}
for port_id, offset_val in dc_offset_map.items():
    machine.ports.analog_outputs["con1"][port_id].offset = float(offset_val)

machine.ports.analog_inputs["con1"][1].gain_db = 20
machine.ports.analog_inputs["con1"][2].gain_db = 20

########################################################################################################################
# %%  4. Mixer corrections (drive & readout)
########################################################################################################################
# Q1 drive mixer
g, phi = correction_matrix_to_gain_phase(lv["Q1_DL_mixer_C_matrix"])
machine.mixers["mixer1_q1.xy"].mixer.correction_gain = g
machine.mixers["mixer1_q1.xy"].mixer.correction_phase = phi

# QC drive mixer (SSB on shared readout ports)
g, phi = correction_matrix_to_gain_phase(lv["QC_mixer_C_matrix"])
machine.mixers["mixer3_qC.xy"].mixer.correction_gain = g
machine.mixers["mixer3_qC.xy"].mixer.correction_phase = phi

# Q2 drive mixer (SSB, but modelled as IQ in QUAM)
g, phi = correction_matrix_to_gain_phase(lv["Q2_DL_mixer_C_matrix"])
machine.mixers["mixer2_q2.xy"].mixer.correction_gain = g
machine.mixers["mixer2_q2.xy"].mixer.correction_phase = phi

# Per-resonator readout mixer corrections
for mixer_name, matrix_key in [
    ("mixer4_q1.rr", "res_1_mixer_C_matrix"),
    ("mixer4_qC.rr", "res_C_mixer_C_matrix"),
    ("mixer4_q2.rr", "res_2_mixer_C_matrix"),
]:
    g, phi = correction_matrix_to_gain_phase(lv[matrix_key])
    machine.mixers[mixer_name].mixer.correction_gain = g
    machine.mixers[mixer_name].mixer.correction_phase = phi

########################################################################################################################
# %%  5. Digital outputs — RF switch control
########################################################################################################################
for qubit in machine.qubits.values():
    qubit.resonator.digital_outputs = {
        "switch_readout": DigitalOutputChannel(
            opx_output=("con1", 1), delay=160, buffer=0
        ),
        "switch_mod": DigitalOutputChannel(
            opx_output=("con1", 2), delay=160, buffer=0
        ),
    }

########################################################################################################################
# %%  6. Qubit gate pulses — DragCosine (cosine window + optional DRAG)
########################################################################################################################
x180_drag_len = [lv["Q1_DL_x180_drag_len"], lv["QC_x180_drag_len"], lv["Q2_DL_x180_drag_len"]]
drag_x180_amp = [lv["Q1_DL_x180_drag_amp"], lv["QC_x180_drag_amp"], lv["Q2_DL_x180_drag_amp"]]
drag_x90_amp = [lv["Q1_DL_x90_drag_amp"], lv["QC_x90_drag_amp"], lv["Q2_DL_x90_drag_amp"]]
drag_coef = [lv.get("Q1_DL_x180_drag_coef", 0.0), lv.get("QC_x180_drag_coef", 0.0), lv.get("Q2_DL_x180_drag_coef", 0.0)]

for k, qubit in enumerate(machine.qubits.values()):
    add_DragCosine_pulses(
        qubit,
        amplitude=float(drag_x180_amp[k]),
        length=int(x180_drag_len[k]),
        anharmonicity=anharmonicity[k],
        alpha=float(drag_coef[k]),
        detuning=0,
    )
    qubit.xy.operations["x90_DragCosine"].amplitude = float(drag_x90_amp[k])

########################################################################################################################
# %%  7. Saturation pulse (long constant pulse for spectroscopy)
########################################################################################################################
for qubit in machine.qubits.values():
    qubit.xy.operations["saturation"].length = 20_000
    qubit.xy.operations["saturation"].amplitude = 0.3

########################################################################################################################
# %%  8. Second flux channel for AC/parametric drive (shares same port as z)
########################################################################################################################
from quam.components.pulses import SquarePulse

z_port_refs = {'qC': "#/ports/analog_outputs/con1/1", 'q2': "#/ports/analog_outputs/con1/2"}
for qubit_name in ['qC', 'q2']:
    z_orig = machine.qubits[qubit_name].z
    z_ac = SingleChannel(
        opx_output=z_port_refs[qubit_name],
        intermediate_frequency=z_orig.intermediate_frequency,
    )
    for op_name, pulse in z_orig.operations.items():
        z_ac.operations[op_name] = SquarePulse(
            length=pulse.length, amplitude=pulse.amplitude
        )
    machine.qubits[qubit_name].z_ac = z_ac

########################################################################################################################
# %%  9. Collect all remaining calibrated values not directly in QUAM
########################################################################################################################
calibrated_extras = {
    "twpa": {
        "freq": lv["twpa_freq"],
        "power": lv["twpa_power"],
        "flux_current": lv["twpa_flux_current"],
    },
    "context_dc_offsets": {
        "DC_offset9_readout": lv["DC_offset9_readout"],
        "DC_offset10_readout": lv["DC_offset10_readout"],
        "DC_offset9_qubit": lv["DC_offset9_qubit"],
        "DC_offset10_qubit": lv["DC_offset10_qubit"],
        "DC_offset7": lv["DC_offset7"],
        "DC_offset8": lv["DC_offset8"],
    },
    "constant_pi_pulses": {
        "Q1_DL_x180_len": lv["Q1_DL_x180_len"],
        "Q1_DL_x180_amp": lv["Q1_DL_x180_amp"],
        "Q2_DL_x180_len": lv["Q2_DL_x180_len"],
        "Q2_DL_x180_amp": lv["Q2_DL_x180_amp"],
        "QC_x180_len": lv["QC_x180_len"],
        "QC_x180_amp": lv["QC_x180_amp"],
    },
    "drag_gaussian_pulses": {
        "Q1_DL_x180_drag_amp": lv["Q1_DL_x180_drag_amp"],
        "Q1_DL_x180_drag_coef": lv["Q1_DL_x180_drag_coef"],
        "Q2_DL_x180_drag_amp": lv["Q2_DL_x180_drag_amp"],
        "Q2_DL_x180_drag_coef": lv["Q2_DL_x180_drag_coef"],
    },
    "ef_transitions": {
        "Q1_ef_freq": lv["Q1_ef_freq"],
        "Q2_ef_freq": lv["Q2_ef_freq"],
        "QC_ef_freq": lv["QC_ef_freq"],
        "Q1_ef_DL_x180_len": lv["Q1_ef_DL_x180_len"],
        "Q1_ef_DL_x180_amp": lv["Q1_ef_DL_x180_amp"],
        "Q2_ef_DL_x180_len": lv["Q2_ef_DL_x180_len"],
        "Q2_ef_DL_x180_amp": lv["Q2_ef_DL_x180_amp"],
        "QC_ef_x180_len": lv["QC_ef_x180_len"],
        "QC_ef_x180_amp": lv["QC_ef_x180_amp"],
    },
    "ef_mixer_corrections": {
        "Q1_ef_DL_mixer_C_matrix": lv["Q1_ef_DL_mixer_C_matrix"],
        "Q2_ef_DL_mixer_C_matrix": lv["Q2_ef_DL_mixer_C_matrix"],
        "QC_ef_mixer_C_matrix": lv["QC_ef_mixer_C_matrix"],
    },
    "parametric_drives": {
        "pd_CZ_amp": lv["pd_CZ_amp"],
        "pd_CZ_freq": lv["pd_CZ_freq"],
        "pd_CZ_square_len": lv["pd_CZ_square_len"],
        "pd_CZ_square_phase_comp_Q1": lv["pd_CZ_square_phase_comp_Q1"],
        "pd_CZ_square_phase_comp_Q2": lv["pd_CZ_square_phase_comp_Q2"],
        "pd_CZ_gaussian_flat_len": lv["pd_CZ_gaussian_flat_len"],
        "pd_CZ_gaussian_flat_phase_comp_Q1": lv["pd_CZ_gaussian_flat_phase_comp_Q1"],
        "pd_CZ_gaussian_flat_phase_comp_Q2": lv["pd_CZ_gaussian_flat_phase_comp_Q2"],
        "pd_CZ_slepian_flat_len": lv["pd_CZ_slepian_flat_len"],
        "pd_CZ_slepian_flat_phase_comp_Q1": lv["pd_CZ_slepian_flat_phase_comp_Q1"],
        "pd_CZ_slepian_flat_phase_comp_Q2": lv["pd_CZ_slepian_flat_phase_comp_Q2"],
        "pd_sqrt_iSWAP_len": lv["pd_sqrt_iSWAP_len"],
        "pd_sqrt_iSWAP_freq": lv["pd_sqrt_iSWAP_freq"],
        "pd_iSWAP_len": lv["pd_iSWAP_len"],
        "pd_iSWAP_freq": lv["pd_iSWAP_freq"],
    },
}

extras_path = Path(__file__).resolve().parent.parent / "calibrated_extras.json"
with open(extras_path, "w") as f:
    json.dump(calibrated_extras, f, indent=4)
print(f"Extra calibrated parameters saved to {extras_path}")

########################################################################################################################
# %%                                         Save the updated QUAM
########################################################################################################################
machine.save()
print("QUAM state saved with calibrated values.")

config = machine.generate_config()
config_path = Path(__file__).resolve().parent.parent / "qua_config.json"
with open(config_path, "w") as f:
    json.dump(config, f, indent=4)
print(f"QUA config written to {config_path}")

########################################################################################################################
# %%                                           Summary
########################################################################################################################
print("\n" + "=" * 72)
print("CALIBRATED PARAMETER SUMMARY")
print("=" * 72)
print(f"JSON source: {JSON_FILE}")
print()
for k, name in enumerate(["q1 (Q1)", "qC (QC)", "q2 (Q2)"]):
    q = list(machine.qubits.values())[k]
    rr_if = float(res_freq[k]) - lo_res
    xy_if = float(qubit_freq[k]) - qubit_lo[k]
    print(f"  {name}:")
    print(f"    f_01         = {qubit_freq[k] / 1e9:.6f} GHz  (IF = {xy_if / 1e6:.3f} MHz)")
    print(f"    anharm       = {anharmonicity[k] / 1e6:.1f} MHz")
    print(f"    res freq     = {res_freq[k] / 1e9:.6f} GHz  (IF = {rr_if / 1e6:.3f} MHz)")
    print(f"    readout amp  = {readout_amp[k]:.6f} V,  len = {readout_len[k]} ns")
    print(f"    drag x180 amp = {drag_x180_amp[k]},  drag x90 amp = {drag_x90_amp[k]}")
    print(f"    DRAG alpha   = {drag_coef[k]}")
    print()
print("DC offsets (V):")
for port_id in [1, 2, 3, 4, 5, 6, 9, 10]:
    print(f"  port {port_id:2d}: {dc_offset_map[port_id]}")
print()
print("Params NOT in QUAM saved to calibrated_extras.json:")
for section in calibrated_extras:
    print(f"  - {section} ({len(calibrated_extras[section])} entries)")
