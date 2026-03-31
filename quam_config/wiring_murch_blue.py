import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from qualang_tools.wirer.wirer.channel_specs import *
from qualang_tools.wirer import Instruments, Connectivity, allocate_wiring, visualize
from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.builder.superconducting import build_quam
from quam_config import Quam, SSBModulatorChannel
from quam.components.hardware import FrequencyConverter, LocalOscillator
from quam.components.ports.analog_outputs import OPXPlusAnalogOutputPort
from quam.components.pulses import SquarePulse

########################################################################################################################
# %%                                              Define static parameters
########################################################################################################################
host_ip = "10.225.208.35"
port = 80
cluster_name = "Cluster_1"

########################################################################################################################
# %%                                      Define the available instrument setup
########################################################################################################################
# Hardware: 1 OPX+ controller (10 analog outputs, 2 analog inputs)
# IQ mixers: 2 (readout + Q2 drive) — registered through the wirer.
# SSB modulators (Q3/Q4 drives) are added manually after build_quam.
#
#   Physical wiring:
#       Port 1      — C23 coupler flux
#       Port 2      — C34 coupler flux
#       Ports 3,4   — Q2 drive, IQ mixer 2
#       Port  5     — Q3 drive, SSB modulator (single output)
#       Ports 7,8   — unused
#       Ports 9,10  — readout IQ mixer 1 + Q4 drive SSB (shared, time-multiplexed)
#       Inputs 1,2  — readout ADC
instruments = Instruments()
instruments.add_opx_plus(controllers=[1])
instruments.add_external_mixer(indices=[1, 2])

########################################################################################################################
# %%                                 Define which qubit ids are present in the system
########################################################################################################################
# 3 fixed-frequency transmons in a linear chain: Q2 -- C23 -- Q3 -- C34 -- Q4
# 2 tunable couplers (C23 between Q2-Q3, C34 between Q3-Q4)
qubits = [2, 3, 4]
qubit_pairs = [(qubits[i], qubits[i + 1]) for i in range(len(qubits) - 1)]

########################################################################################################################
# %%                                 Define any custom/hardcoded channel addresses
########################################################################################################################
# Only readout and Q2 drive go through the wirer (IQ mixers).
# Q3 and Q4 SSB drives are created manually after build_quam.
res_ch = opx_iq_ext_mixer_spec(in_port_i=1, in_port_q=2, out_port_i=9, out_port_q=10, mixer_index=1)
q2_drive_ch = opx_iq_ext_mixer_spec(out_port_i=3, out_port_q=4, mixer_index=2)
c23_flux_ch = opx_spec(out_port=1)
c34_flux_ch = opx_spec(out_port=2)

########################################################################################################################
# %%                 Allocate the wiring to the connectivity object based on the available instruments
########################################################################################################################
connectivity = Connectivity()
connectivity.add_resonator_line(qubits=qubits, constraints=res_ch)
connectivity.add_qubit_drive_lines(qubits=[2], constraints=q2_drive_ch)
connectivity.add_qubit_pair_flux_lines(qubit_pairs=[(2, 3)], constraints=c23_flux_ch)
connectivity.add_qubit_pair_flux_lines(qubit_pairs=[(3, 4)], constraints=c34_flux_ch)
allocate_wiring(connectivity, instruments)

# View wiring schematic
visualize(connectivity.elements, available_channels=instruments.available_channels)
plt.show(block=True)

########################################################################################################################
# %%                                   Build the wiring and QUAM
########################################################################################################################
user_input = input("Do you want to save the updated QUAM? (y/n)")
if user_input.lower() == "y":
    machine = Quam()
    build_quam_wiring(connectivity, host_ip, cluster_name, machine)

    machine = Quam.load()
    build_quam(machine)

    # --- Q3 and Q4 SSB modulators (not supported by the wirer) ---
    # Create port 5 for Q3 (port 9 already exists from readout allocation).
    machine.ports.analog_outputs["con1"][5] = OPXPlusAnalogOutputPort(
        controller_id="con1", port_id=5
    )

    # Frequency converters for the SSB drives (LO only, no IQ mixer corrections).
    machine.mixers["ssb_q3.xy"] = FrequencyConverter(
        local_oscillator=LocalOscillator(),
    )
    machine.mixers["ssb_q4.xy"] = FrequencyConverter(
        local_oscillator=LocalOscillator(),
    )

    # Attach SSBModulator channels to Q3 and Q4.
    # Q3 SSB on port 5, Q4 SSB on port 9 (shared with readout, time-multiplexed).
    machine.qubits["q3"].xy = SSBModulatorChannel(
        opx_output="#/ports/analog_outputs/con1/5",
        frequency_converter_up="#/mixers/ssb_q3.xy",
        operations={"saturation": SquarePulse(length=20000, amplitude=0.25)},
    )
    machine.qubits["q4"].xy = SSBModulatorChannel(
        opx_output="#/ports/analog_outputs/con1/9",
        frequency_converter_up="#/mixers/ssb_q4.xy",
        operations={"saturation": SquarePulse(length=20000, amplitude=0.25)},
    )

    machine.save()
