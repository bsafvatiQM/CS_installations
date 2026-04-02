import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
print(sys.executable)
import matplotlib.pyplot as plt
from qualang_tools.wirer.wirer.channel_specs import *
from qualang_tools.wirer import Instruments, Connectivity, allocate_wiring, visualize
from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.builder.superconducting import build_quam
from quam_config import Quam

########################################################################################################################
# %%                                              Define static parameters
########################################################################################################################
host_ip = '10.225.208.230'
port = 80
cluster_name = "Cluster_1"

########################################################################################################################
# %%                                      Define the available instrument setup
########################################################################################################################
# Hardware: 1 OPX+ controller (10 analog outputs, 2 analog inputs)

instruments = Instruments()
instruments.add_opx_plus(controllers=[1])
instruments.add_external_mixer(indices=[1, 2, 3, 4])

########################################################################################################################
# %%                                 Define which qubit ids are present in the system
########################################################################################################################


qubits = [1, "C", 2]

########################################################################################################################
# %%                                 Define any custom/hardcoded channel addresses
########################################################################################################################
# All port constraints match the physical cabling described in sample_config_black.py.
res_ch = opx_iq_ext_mixer_spec(in_port_i=2, in_port_q=1, out_port_i=9, out_port_q=10, mixer_index=4)
Q1_drive_ch = opx_iq_ext_mixer_spec(out_port_i=3, out_port_q=4, mixer_index=1)
# Q3 physically shares ports 9,10 with readout; allocate on 7,8 as placeholder (patched below).
QC_drive_ch = opx_iq_ext_mixer_spec(out_port_i=9, out_port_q=10 , mixer_index=3)
Q2_drive_ch = opx_iq_ext_mixer_spec(out_port_i=5, out_port_q=6, mixer_index=2)
QC_flux_ch = opx_spec(out_port=1)
Q2_flux_ch = opx_spec(out_port=2)

########################################################################################################################
# %%                 Allocate the wiring to the connectivity object based on the available instruments
########################################################################################################################
connectivity = Connectivity()
connectivity.add_resonator_line(qubits=qubits, constraints=res_ch)
allocate_wiring(connectivity, instruments,block_used_channels=False)
connectivity.add_qubit_drive_lines(qubits=[1], constraints=Q1_drive_ch)
connectivity.add_qubit_drive_lines(qubits=[2], constraints=Q2_drive_ch)
connectivity.add_qubit_drive_lines(qubits=["C"], constraints=QC_drive_ch)
allocate_wiring(connectivity, instruments,block_used_channels=False)
connectivity.add_qubit_flux_lines(qubits=[2], constraints=Q2_flux_ch)
connectivity.add_qubit_flux_lines(qubits=["C"], constraints=QC_flux_ch)
allocate_wiring(connectivity, instruments,block_used_channels=False)

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
    # # Patch Q3 drive from placeholder ports 7,8 to the real shared ports 9,10.
    # # Physically, Q3's SSB modulator shares the readout IQ mixer's OPX ports,
    # # time-multiplexed via digital switches (matches QC in sample_config_black.py).
    # machine.wiring["qubits"]["q3"]["xy"]["opx_output_I"] = "#/ports/analog_outputs/con1/9"
    # machine.wiring["qubits"]["q3"]["xy"]["opx_output_Q"] = "#/ports/analog_outputs/con1/10"

    build_quam(machine)
