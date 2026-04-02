import json
import numpy as np
import h5py
from scipy.signal.windows import chebwin, taylor, flattop
from scipy.fft import fft, fftfreq
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

json_file = r'C:/shared-root/users/Emily/latest_values_xiayu11.json'
with open(json_file, 'r') as f:
    latest_values = json.load(f)

"""OPX controller name"""
cname = 'con1'
qop_ip = '10.225.208.230'
port = 80

"""These are the res settings"""
# Frequencies
lo_res = 6982000000
# lo_res = 6882000000

if_res_1 = latest_values['res_1_freq'] - lo_res
if_res_2 = latest_values['res_2_freq'] - lo_res
if_res_C = latest_values['res_C_freq'] - lo_res

# TWPA
twpa_freq = latest_values['twpa_freq']
twpa_power = latest_values['twpa_power']
twpa_flux_current = latest_values['twpa_flux_current']

# Readout pulse
time_readout_res_1 = latest_values['time_readout_res_1']
time_readout_res_2 = latest_values['time_readout_res_2']
time_readout_res_C = latest_values['time_readout_res_C']
amp_readout_res_1 = latest_values['amp_readout_res_1']
amp_readout_res_2 = latest_values['amp_readout_res_2']
amp_readout_res_C = latest_values['amp_readout_res_C']


readout_wf_t = np.arange(16000)

readout_wf_res_1 = np.full_like(readout_wf_t, fill_value=amp_readout_res_1, dtype=np.float64)
readout_wf_res_2 = np.full_like(readout_wf_t, fill_value=amp_readout_res_2, dtype=np.float64)
readout_wf_res_C = np.full_like(readout_wf_t, fill_value=amp_readout_res_C, dtype=np.float64)

# intervals = [0, 3200, 16000]  # Including the start and end
# # Use np.digitize to map x values to interval indices
# indices = np.digitize(readout_wf_t, bins=intervals) - 1  # Subtract 1 to zero-index
# readout_wf_res_1 = np.array(amp_readout_res_1)[indices]
# readout_wf_res_2 = np.array(amp_readout_res_2)[indices]
# readout_wf_res_C = np.array(amp_readout_res_C)[indices]

integration_weights_file = r'C:/shared-root/users/Emily/Data/integration_weights.hdf5'
with h5py.File(integration_weights_file, 'r') as hf:
    durations = np.array(hf['durations'][:])
    res_1_iw_I = np.array(hf['res_1_iw_I'][:])
    res_1_iw_Q = np.array(hf['res_1_iw_Q'][:])
    res_2_iw_I = np.array(hf['res_2_iw_I'][:])
    res_2_iw_Q = np.array(hf['res_2_iw_Q'][:])

res_C_iw_I = res_1_iw_I
res_C_iw_Q = res_1_iw_Q

save_load_dir = r'C:/shared-root/users/Emily/Data/'


"""These are the qubit settings"""
def sine_w(length, f_offset):
    n = np.arange(length)
    N = length - 1
    n_offset = n - N / 2
    return np.sin((np.pi / N) * n) * np.exp((2j * np.pi * f_offset) * n_offset)


def gen_sqisw_12_wf(amp_QC_ffl, freq_QC_ffl, sqisw_12_len, sqisw_12_dc, sqisw_12_taylor_M, sqisw_12_taylor_nbar, sqisw_12_taylor_sll):
    t = np.arange(sqisw_12_len) - 0.5 * (sqisw_12_len - 1)
    taylor_win = taylor(sqisw_12_taylor_M, sqisw_12_taylor_nbar, sqisw_12_taylor_sll)
    taylor_win /= np.sum(taylor_win)
    square_wf = np.full(sqisw_12_len - sqisw_12_taylor_M + 1, 1.0)
    win = np.convolve(square_wf, taylor_win)
    sqisw_12_wf = sqisw_12_dc + amp_QC_ffl * np.cos(2 * np.pi * freq_QC_ffl * t * 1e-9)
    sqisw_12_wf = win * sqisw_12_wf
    return sqisw_12_wf

# --- DRAG helpers (uses qualang_tools if available; otherwise a safe fallback) ---
try:
    from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms
    _USE_QTOOLS = True
except Exception:
    _USE_QTOOLS = False
    def drag_gaussian_pulse_waveforms(amp, length, sigma, alpha, anharm_hz, ac_stark_hz):
        """
        Fallback generator: gaussian (I) and alpha * derivative-of-gaussian (Q).
        Units: length, sigma in samples (ns at 1 GS/s), anharm/ac_stark in Hz (not used in fallback).
        alpha is dimensionless and will be calibrated on hardware.
        """
        t = np.arange(length) - 0.5 * (length - 1)
        g = amp * np.exp(-(t**2) / (2.0 * sigma**2))
        dg = -(t / (sigma**2)) * g
        return np.array([g, alpha * dg], dtype=object)

# Frequencies
# lo_qubit = 4099000000
lo_qubit = 4241000000
# lo_mod = 5197000000
lo_mod = 5347000000

Q1_freq = latest_values['Q1_freq']
Q2_freq = latest_values['Q2_freq']
QC_freq = latest_values['QC_freq']

Q1_ef_freq = latest_values['Q1_ef_freq']
Q2_ef_freq = latest_values['Q2_ef_freq']
QC_ef_freq = latest_values['QC_ef_freq']

if_Q1 = latest_values['Q1_freq'] - lo_qubit
if_Q2 = latest_values['Q2_freq'] - lo_qubit
if_QC = latest_values['QC_freq'] - lo_mod

if_ACS = latest_values['ACS_freq'] - lo_qubit

if_Q1_ef = latest_values['Q1_ef_freq'] - lo_qubit
if_Q2_ef = latest_values['Q2_ef_freq'] - lo_qubit
if_QC_ef = latest_values['QC_ef_freq'] - lo_mod

# Pi-pulse
Q1_DL_x180_len = latest_values['Q1_DL_x180_len']
Q1_DL_x180_amp = latest_values['Q1_DL_x180_amp']
Q1_DL_x90_len = Q1_DL_x180_len // 2
Q1_DL_x90_amp = Q1_DL_x180_amp
Q1_DL_x180_long_amp = latest_values['Q1_DL_x180_long_amp']
Q2_DL_x180_long_amp = latest_values['Q2_DL_x180_long_amp']
QC_x180_long_amp = latest_values['QC_x180_long_amp']


Q2_DL_x180_len = latest_values['Q2_DL_x180_len']
Q2_DL_x180_amp = latest_values['Q2_DL_x180_amp']
Q2_DL_x90_len = Q2_DL_x180_len // 2
Q2_DL_x90_amp = Q2_DL_x180_amp

QC_x180_len = latest_values['QC_x180_len']
QC_x180_amp = latest_values['QC_x180_amp']
QC_x90_len = QC_x180_len // 2
QC_x90_amp = QC_x180_amp

Q1_ef_DL_x180_len = latest_values['Q1_ef_DL_x180_len']
Q1_ef_DL_x180_amp = latest_values['Q1_ef_DL_x180_amp']
Q1_ef_DL_x90_len = Q1_ef_DL_x180_len // 2
Q1_ef_DL_x90_amp = Q1_ef_DL_x180_amp

Q2_ef_DL_x180_len = latest_values['Q2_ef_DL_x180_len']
Q2_ef_DL_x180_amp = latest_values['Q2_ef_DL_x180_amp']
Q2_ef_DL_x90_len = Q2_ef_DL_x180_len // 2
Q2_ef_DL_x90_amp = Q2_ef_DL_x180_amp

QC_ef_x180_len = latest_values['QC_ef_x180_len']
QC_ef_x180_amp = latest_values['QC_ef_x180_amp']
QC_ef_x90_len = QC_ef_x180_len // 2
QC_ef_x90_amp = QC_ef_x180_amp

x180_cos_len = 88
x90_cos_len = 44
x180_cos_wf = sine_w(x180_cos_len, 0)
x90_cos_wf = sine_w(x90_cos_len, 0)

Q1_DL_x180_drag_len = Q1_DL_x180_len
Q1_DL_x180_drag_amp = latest_values['Q1_DL_x180_drag_amp']
Q1_DL_x180_drag_coef = latest_values['Q1_DL_x180_drag_coef']

Q2_DL_x180_drag_len = Q2_DL_x180_len
Q2_DL_x180_drag_amp = latest_values['Q2_DL_x180_drag_amp']
Q2_DL_x180_drag_coef = latest_values['Q2_DL_x180_drag_coef']

Q1_DL_x180_cos_amp = latest_values['Q1_DL_x180_cos_amp']
Q1_DL_x90_cos_amp = latest_values['Q1_DL_x90_cos_amp']
Q2_DL_x180_cos_amp = latest_values['Q2_DL_x180_cos_amp']
Q2_DL_x90_cos_amp = latest_values['Q2_DL_x90_cos_amp']
QC_x180_cos_amp = latest_values['QC_x180_cos_amp']
QC_x90_cos_amp = latest_values['QC_x90_cos_amp']


x180_ef_cos_len = 64
x90_ef_cos_len = 32
x180_ef_cos_wf = sine_w(x180_ef_cos_len, 0)
x90_ef_cos_wf = sine_w(x90_ef_cos_len, 0)

Q1_ef_DL_x180_cos_amp = latest_values['Q1_ef_DL_x180_cos_amp']
Q1_ef_DL_x90_cos_amp = latest_values['Q1_ef_DL_x90_cos_amp']
Q2_ef_DL_x180_cos_amp = latest_values['Q2_ef_DL_x180_cos_amp']
Q2_ef_DL_x90_cos_amp = latest_values['Q2_ef_DL_x90_cos_amp']
QC_ef_x180_cos_amp = latest_values['QC_ef_x180_cos_amp']
QC_ef_x90_cos_amp = latest_values['QC_ef_x90_cos_amp']

amp_ACS = latest_values['amp_ACS']
QC_FFL_amp = latest_values['QC_FFL_amp']
QC_FFL_freq = latest_values['QC_FFL_freq']
QC_FFL_len = 4 * round(latest_values['QC_FFL_len_mod_4'])
QC_FFL_frame_rot = latest_values['QC_FFL_frame_rot']

pd_CZ_len = latest_values['pd_CZ_len']
pd_CZ_freq = latest_values['pd_CZ_freq']
pd_sqrt_iSWAP_len= latest_values['pd_sqrt_iSWAP_len']
pd_sqrt_iSWAP_freq = latest_values['pd_sqrt_iSWAP_freq']
pd_iSWAP_len= latest_values['pd_iSWAP_len']
pd_iSWAP_freq = latest_values['pd_iSWAP_freq']

#DC offsets
DC_offset1 = latest_values['DC_offset1']
DC_offset2 = latest_values['DC_offset2']
DC_offset3 = latest_values['DC_offset3']
DC_offset4 = latest_values['DC_offset4']
DC_offset5 = latest_values['DC_offset5']
DC_offset6 = latest_values['DC_offset6']
DC_offset7 = latest_values['DC_offset7']
DC_offset8 = latest_values['DC_offset8']
DC_offset9 = latest_values['DC_offset9']
DC_offset10 = latest_values['DC_offset10']
DC_offset9_readout = latest_values['DC_offset9_readout']
DC_offset10_readout = latest_values['DC_offset10_readout']
DC_offset9_qubit = latest_values['DC_offset9_qubit']
DC_offset10_qubit = latest_values['DC_offset10_qubit']

DC_offset_input1 = 0.0
DC_offset_input2 = 0.0


# Mixer correction matrices
Q1_DL_mixer_C_matrix = latest_values['Q1_DL_mixer_C_matrix']
Q2_DL_mixer_C_matrix = latest_values['Q2_DL_mixer_C_matrix']
QC_mixer_C_matrix = latest_values['QC_mixer_C_matrix']
Q1_ef_DL_mixer_C_matrix = latest_values['Q1_ef_DL_mixer_C_matrix']
Q2_ef_DL_mixer_C_matrix = latest_values['Q2_ef_DL_mixer_C_matrix']
QC_ef_mixer_C_matrix = latest_values['QC_ef_mixer_C_matrix']
res_1_mixer_C_matrix = latest_values['res_1_mixer_C_matrix']
res_2_mixer_C_matrix = latest_values['res_2_mixer_C_matrix']
res_C_mixer_C_matrix = latest_values['res_C_mixer_C_matrix']
ACS_mixer_C_matrix = latest_values['ACS_mixer_C_matrix']

# ---- DRAG parameters for Q1 (GE) ----
# length already defined in your file:
#   Q1_DL_x180_drag_len, Q1_DL_x180_drag_amp
Q1_DL_sigma = Q1_DL_x180_drag_len // 4                  # standard choice; keep multiple of 4
assert Q1_DL_x180_drag_len % 4 == 0, "Q1_DL_x180_drag_len must be a multiple of 4 ns"

# Anharmonicity from your latest_values (Hz). Usually negative.
anharm_Q1_hz = float(Q1_ef_freq - Q1_freq)

# Optional AC-Stark detuning during the pulse (Hz). Keep 0 unless you purposely detune.
AC_stark_Q1_hz = float(latest_values.get("Q1_ACS_detune_hz", 0.0))

# DRAG coefficient (dimensionless). Start at 0.0 and calibrate later.
drag_coef_Q1 = float(latest_values.get("Q1_DL_x180_drag_coef", 0.0))

# ---- Build the x180 DRAG waveforms (I: gaussian, Q: derivative scaled by alpha) ----
x180_drag_I_Q1, x180_drag_Q_Q1 = np.array(
    drag_gaussian_pulse_waveforms(
        Q1_DL_x180_drag_amp,      # amplitude
        Q1_DL_x180_drag_len,      # length (ns)
        Q1_DL_sigma,              # sigma (ns)
        Q1_DL_x180_drag_coef,     # DRAG alpha (dimensionless)
        anharm_Q1_hz,             # anharmonicity (Hz)
        AC_stark_Q1_hz            # AC-Stark detuning (Hz)
    ),
    dtype=object
)

# length already defined in your file:
#   Q2_DL_x180_drag_len, Q2_DL_x180_drag_amp
Q2_DL_sigma = Q2_DL_x180_drag_len // 4                  # standard choice; keep multiple of 4
assert Q2_DL_x180_drag_len % 4 == 0, "Q2_DL_x180_drag_len must be a multiple of 4 ns"

# Anharmonicity from your latest_values (Hz). Usually negative.
anharm_Q2_hz = float(Q2_ef_freq - Q2_freq)

# Optional AC-Stark detuning during the pulse (Hz). Keep 0 unless you purposely detune.
AC_stark_Q2_hz = float(latest_values.get("Q2_ACS_detune_hz", 0.0))

# DRAG coefficient (dimensionless). Start at 0.0 and calibrate later.
drag_coef_Q2 = float(latest_values.get("Q2_DL_x180_drag_coef", 0.0))

# ---- Build the x180 DRAG waveforms (I: gaussian, Q: derivative scaled by alpha) ----
x180_drag_I_Q2, x180_drag_Q_Q2 = np.array(
    drag_gaussian_pulse_waveforms(
        Q2_DL_x180_drag_amp,      # amplitude
        Q2_DL_x180_drag_len,      # length (ns)
        Q2_DL_sigma,              # sigma (ns)
        Q2_DL_x180_drag_coef,     # DRAG alpha (dimensionless)
        anharm_Q2_hz,             # anharmonicity (Hz)
        AC_stark_Q2_hz            # AC-Stark detuning (Hz)
    ),
    dtype=object
)


"""Below is the config_std that is imported to scripts"""
config = {
    'version': 1,
    'controllers': {
        cname: {
            'type': cname,
            'analog_outputs': {
                1: {'offset': DC_offset1},
                2: {'offset': DC_offset2},
                3: {'offset': DC_offset3},
                4: {'offset': DC_offset4},
                5: {'offset': DC_offset5},
                6: {'offset': DC_offset6},
                7: {'offset': DC_offset7},
                8: {'offset': DC_offset8},
                9: {'offset': DC_offset9},
                10: {'offset': DC_offset10}
            },
            'digital_outputs': {
                1: {},
                2: {},
                3: {}
            },
            'analog_inputs': {
                1: {'gain_db': 20, 'offset': DC_offset_input1},
                2: {'gain_db': 20, 'offset': DC_offset_input2},
            }
        },
    },
    'elements': {
        'res_1': {
            'mixInputs': {
                'I': (cname, 9),
                'Q': (cname, 10),
                'mixer': 'mixer_readout_line',
                'lo_frequency': lo_res
            },
            'digitalInputs': {
                'switch_readout': {
                    'buffer': 0,
                    'delay': 160,
                    'port': (cname, 1)
                },
                'switch_mod': {
                    'buffer': 0,
                    'delay': 160,
                    'port': (cname, 2)
                }
            },
            'thread': 'Q1',
            'intermediate_frequency': if_res_1,
            'operations': {
                'square': 'square_pulse',
                'readout_16us': 'readout_16us_pulse_res_1',
                'readout_4us': 'readout_4us_pulse_res_1',
                'readout_5us': 'readout_5us_pulse_res_1',
                'readout_fast': 'readout_fast_pulse_res_1',
                'readout_calib': 'readout_calib_pulse_res_1'
            },
            'time_of_flight': 24,
            'smearing': 0,
            'outputs': {
                'out1': (cname, 1),
                'out2': (cname, 2)
            }
        },
        'res_2': {
            'mixInputs': {
                'I': (cname, 9),
                'Q': (cname, 10),
                'mixer': 'mixer_readout_line',
                'lo_frequency': lo_res
            },
            'digitalInputs': {
                'switch_readout': {
                    'buffer': 0,
                    'delay': 160,
                    'port': (cname, 1)
                },
                'switch_mod': {
                    'buffer': 0,
                    'delay': 160,
                    'port': (cname, 2)
                }
            },
            'thread': 'Q2',
            'intermediate_frequency': if_res_2,
            'operations': {
                'square': 'square_pulse',
                'readout_16us': 'readout_16us_pulse_res_2',
                'readout_4us': 'readout_4us_pulse_res_2',
                'readout_5us': 'readout_5us_pulse_res_2',
                'readout_fast': 'readout_fast_pulse_res_2',
                'readout_calib': 'readout_calib_pulse_res_2'
            },
            'time_of_flight': 24,
            'smearing': 0,
            'outputs': {
                'out1': (cname, 1),
                'out2': (cname, 2)
            }
        },
        'res_C': {
            'mixInputs': {
                'I': (cname, 9),
                'Q': (cname, 10),
                'mixer': 'mixer_readout_line',
                'lo_frequency': lo_res
            },
            'digitalInputs': {
                'switch_readout': {
                    'buffer': 0,
                    'delay': 160,
                    'port': (cname, 1)
                },
                'switch_mod': {
                    'buffer': 0,
                    'delay': 160,
                    'port': (cname, 2)
                }
            },
            'thread': 'QC',
            'intermediate_frequency': if_res_C,
            'operations': {
                'square': 'square_pulse',
                'readout_16us': 'readout_16us_pulse_res_C',
                'readout_4us': 'readout_4us_pulse_res_C',
                'readout_fast': 'readout_fast_pulse_res_C',
                'readout_calib': 'readout_calib_pulse_res_C'
            },
            'time_of_flight': 24,
            'smearing': 0,
            'outputs': {
                'out1': (cname, 1),
                'out2': (cname, 2)
            }
        },
        'Q1': {
            'mixInputs': {
                'I': (cname, 9),
                'Q': (cname, 10),
                'lo_frequency': lo_qubit
            },
            'digitalInputs': {
                'switch_qubit': {
                    'buffer': 0,
                    'delay': 136,
                    'port': (cname, 3)
                },
                'switch_mod': {
                    'buffer': 0,
                    'delay': 136,
                    'port': (cname, 2)
                }
            },
            'intermediate_frequency': if_Q1,
            'thread': 'Q1',
            'operations': {
                'square': 'square_pulse_Q1',
                'x180': 'x180_pulse_Q1',
                'x90': 'x90_pulse_Q1',
                'x180_cos': 'x180_cos_pulse_Q1',
                'x90_cos': 'x90_cos_pulse_Q1'
            }
        },
        'Q2': {
            'mixInputs': {
                'I': (cname, 9),
                'Q': (cname, 10),
                'lo_frequency': lo_qubit
            },
            'digitalInputs': {
                'switch_qubit': {
                    'buffer': 0,
                    'delay': 136,
                    'port': (cname, 3)
                },
                'switch_mod': {
                    'buffer': 0,
                    'delay': 136,
                    'port': (cname, 2)
                }
            },
            'intermediate_frequency': if_Q2,
            'thread': 'Q2',
            'operations': {
                'square': 'square_pulse',
                'x180': 'x180_pulse_Q2',
                'x90': 'x90_pulse_Q2'
            }
        },
        'ACS': {
            'mixInputs': {
                'I': (cname, 9),
                'Q': (cname, 10),
                'mixer': 'mixer_readout_line',
                'lo_frequency': lo_qubit,
            },
            'digitalInputs': {
                'switch_qubit': {
                    'buffer': 0,
                    'delay': 136,
                    'port': (cname, 3)
                },
                'switch_mod': {
                    'buffer': 136,
                    'delay': 136,
                    'port': (cname, 2)
                }
            },
            'intermediate_frequency': if_ACS,
            'thread': 'ACS',
            'operations': {
                'square': 'square_pulse',
                'square_pad': 'square_pad_pulse'
            }
        },
        # 'QC': {
        #     'mixInputs': {
        #         'I': (cname, 3),
        #         'Q': (cname, 4),
        #         'lo_frequency': lo_qubit,
        #         'mixer': 'mixer_drive_line1',
        #     },
        #     'digitalInputs': {
        #         'switch': {
        #             'buffer': 0,
        #             'delay': 0,
        #             'port': (cname, 3)
        #         }
        #     },
        #     'intermediate_frequency': if_QC,
        #     'thread': 'QC',
        #     'operations': {
        #         'square': 'square_pulse',
        #         'x180': 'x180_pulse_QC',
        #         'x90': 'x90_pulse_QC',
        #         'x180_cos': 'x180_cos_pulse_QC',
        #         'x90_cos': 'x90_cos_pulse_QC'
        #     }
        # },
        'QC': {
            'mixInputs': {
                'I': (cname, 9),
                'Q': (cname, 10),
                'mixer': 'mixer_readout_line',
                'lo_frequency': lo_mod,
            },
            'intermediate_frequency': if_QC,
            'thread': 'QC',
            'operations': {
                'square': 'square_pulse',
                'x180': 'x180_pulse_QC',
                'x90': 'x90_pulse_QC',
                'x180_cos': 'x180_cos_pulse_QC',
                'x90_cos': 'x90_cos_pulse_QC'
            }
        },
        'QC_ef': {
            'mixInputs': {
                'I': (cname, 9),
                'Q': (cname, 10),
                'mixer': 'mixer_readout_line',
                'lo_frequency': lo_mod,
            },
            'intermediate_frequency': if_QC_ef,
            'thread': 'QC',
            'operations': {
                'square': 'square_pulse',
                'x180': 'x180_pulse_QC_ef',
                'x90': 'x90_pulse_QC_ef',
                'x180_cos': 'x180_cos_pulse_QC_ef',
                'x90_cos': 'x90_cos_pulse_QC_ef'
            }
        },
        'Q1_ef': {
            'mixInputs': {
                'I': (cname, 3),
                'Q': (cname, 4)
            },
            'digitalInputs': {
                'switch': {
                    'buffer': 0,
                    'delay': 0,
                    'port': (cname, 3)
                }
            },
            'intermediate_frequency': if_Q1_ef,
            'thread': 'Q1',
            'operations': {
                'square': 'square_pulse',
                'x180': 'x180_pulse_Q1_DL',
                'x90': 'x90_pulse_Q1_DL',
                'x180_cos': 'x180_cos_pulse_Q1_DL',
                'x90_cos': 'x90_cos_pulse_Q1_DL'
            }
        },
        'Q2_ef': {
            'mixInputs': {
                'I': (cname, 3),
                'Q': (cname, 4)
            },
            'digitalInputs': {
                'switch': {
                    'buffer': 0,
                    'delay': 0,
                    'port': (cname, 3)
                }
            },
            'intermediate_frequency': if_Q2_ef,
            'thread': 'Q2',
            'operations': {
                'square': 'square_pulse'
            }
        },
        # 'QC_ef': {
        #     'mixInputs': {
        #         'I': (cname, 3),
        #         'Q': (cname, 4)
        #     },
        #     'digitalInputs': {
        #         'switch': {
        #             'buffer': 0,
        #             'delay': 0,
        #             'port': (cname, 3)
        #         }
        #     },
        #     'intermediate_frequency': if_QC_ef,
        #     'thread': 'QC',
        #     'operations': {
        #         'square': 'square_pulse',
        #         'x180': 'x180_pulse_QC_ef',
        #         'x90': 'x90_pulse_QC_ef',
        #         'x180_cos': 'x180_cos_pulse_QC_ef',
        #         'x90_cos': 'x90_cos_pulse_QC_ef'
        #     }
        # },
        'Q1_DL': {
            'mixInputs': {
                'I': (cname, 3),
                'Q': (cname, 4),
                'mixer': 'mixer_Q1_DL',
                'lo_frequency': lo_qubit
            },
            'intermediate_frequency': if_Q1,
            'thread': 'Q1',
            'operations': {
                'square': 'square_pulse',
                'square_pad': 'square_pad_pulse',
                'x180': 'x180_pulse_Q1_DL',
                'x360': 'x360_pulse_Q1_DL',
                'x90': 'x90_pulse_Q1_DL',
                'x180_cos': 'x180_cos_pulse_Q1_DL',
                'x90_cos': 'x90_cos_pulse_Q1_DL',
                'x180_drag': 'x180_drag_pulse_Q1_DL'
            }
        },
        'Q1_ef_DL': {
            'mixInputs': {
                'I': (cname, 3),
                'Q': (cname, 4),
                'mixer': 'mixer_Q1_ef_DL',
                'lo_frequency': lo_qubit
            },
            'intermediate_frequency': if_Q1_ef,
            'thread': 'Q1',
            'operations': {
                'square': 'square_pulse',
                'x180': 'x180_pulse_Q1_ef_DL',
                'x90': 'x90_pulse_Q1_ef_DL',
                'x180_cos': 'x180_cos_pulse_Q1_ef_DL',
                'x90_cos': 'x90_cos_pulse_Q1_ef_DL'
            }
        },
        'Q2_DL': {
            'mixInputs': {
                'I': (cname, 5),
                'Q': (cname, 6),
                'mixer': 'mixer_Q2_DL',
                'lo_frequency': lo_qubit
            },
            'intermediate_frequency': if_Q2,
            'thread': 'Q2',
            'operations': {
                'square': 'square_pulse',
                'square_pad': 'square_pad_pulse',
                'x180': 'x180_pulse_Q2_DL',
                'x90': 'x90_pulse_Q2_DL',
                'x180_cos': 'x180_cos_pulse_Q2_DL',
                'x90_cos': 'x90_cos_pulse_Q2_DL',
                'x180_drag': 'x180_drag_pulse_Q2_DL'
            }
        },
        'Q2_ef_DL': {
            'mixInputs': {
                'I': (cname, 5),
                'Q': (cname, 6),
                'mixer': 'mixer_Q2_ef_DL',
                'lo_frequency': lo_qubit
            },
            'intermediate_frequency': if_Q2_ef,
            'thread': 'Q2',
            'operations': {
                'square': 'square_pulse',
                'x180': 'x180_pulse_Q2_ef_DL',
                'x90': 'x90_pulse_Q2_ef_DL',
                'x180_cos': 'x180_cos_pulse_Q2_ef_DL',
                'x90_cos': 'x90_cos_pulse_Q2_ef_DL'
            }
        },
        'Q2_DL_HF': {
            'mixInputs': {
                'I': (cname, 7),
                'Q': (cname, 8)
            }
        },
        'dc_offset_Q2': {
            'singleInput': {
                'port': (cname, 2)
            }
        },
        'dc_offset_QC': {
            'singleInput': {
                'port': (cname, 1)
            }
        },
        'pd_1C': {
            'singleInput': {
                'port': (cname, 1)
            },
            'intermediate_frequency': 404000000,
            'operations': {
                'square': 'square_pulse_pd_1C',
                'sqisw': 'sqisw_pulse_pd_1C'
            }
        },
        'ext_flux': {
            'singleInput': {
                'port': (cname, 7)
            }
        },
        'pd_12': {
            'singleInput': {
                'port': (cname, 1)
            },
            'intermediate_frequency': QC_FFL_freq,
            'thread': 'Q1',
            'operations': {
                'square': 'square_pulse_pd_12'
            }
        },
        'QC_FFL': {
            'mixInputs': {
                'I': (cname, 1),
                'Q': (cname, 7),
                'mixer': 'mixer_QC_FFL_dummy',
                'lo_frequency': 0
            },
            'intermediate_frequency': (Q2_freq - Q1_freq) // 2,
            'operations': {
                'square': 'square_pulse_QC_FFL',
            }
        },
        'pd_CZ': {
            'singleInput': {
                'port': (cname, 1),
            },
            'intermediate_frequency': pd_CZ_freq,
            'operations': {
                'square': 'square_pulse_pd_CZ'
            }
        },
        'pd_sqrt_iSWAP': {
            'singleInput': {
                'port': (cname, 1),
            },
            'intermediate_frequency': pd_sqrt_iSWAP_freq,
            'operations': {
                'square': 'square_pulse_pd_sqrt_iSWAP'
            }
        },
        'pd_iSWAP': {
            'singleInput': {
                'port': (cname, 1),
            },
            'intermediate_frequency': pd_iSWAP_freq,
            'operations': {
                'square': 'square_pulse_pd_iSWAP'
            }
        },
    },
    'pulses': {
        'readout_16us_pulse_res_1': {
            'operation': 'measurement',
            'length': 16000,
            'waveforms': {
                'I': 'readout_wf_res_1',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'calib_3200ns': 'calib_3200ns_weights',
                'calib_16us': 'calib_16us_weights',
                'I_cos': 'res_1_iw_I_cosine',
                'I_sin': 'res_1_iw_I_sine',
                'J_cos': 'res_1_iw_J_cosine',
                'J_sin': 'res_1_iw_J_sine',
                'K_cos': 'res_1_iw_K_cosine',
                'K_sin': 'res_1_iw_K_sine',
                'L_cos': 'res_1_iw_L_cosine',
                'L_sin': 'res_1_iw_L_sine',
            }
        },
        'readout_4us_pulse_res_1': {
            'operation': 'measurement',
            'length': 4000,
            'waveforms': {
                'I': 'readout_4us_wf_res_1',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'readout_4us': 'readout_4us_weights',
                "cos": "cosine_weights_4us",
                "sin": "sine_weights_4us",
                "minus_sin": "minus_sine_weights_4us",
            }
        },
        'readout_5us_pulse_res_1': {
            'operation': 'measurement',
            'length': 5000,
            'waveforms': {
                'I': 'readout_5us_wf_res_1',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'readout_5us': 'readout_5us_weights',
                "cos": "cosine_weights",
                "sin": "sine_weights",
                "minus_sin": "minus_sine_weights",
            }
        },
        'readout_calib_pulse_res_1': {
            'operation': 'measurement',
            'length': 16000,
            'waveforms': {
                'I': 'readout_wf_res_1',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'calib_3200ns': 'calib_3200ns_weights',
                'calib_16us': 'calib_16us_weights'
            }
        },
        'readout_fast_pulse_res_1': {
            'operation': 'measurement',
            'length': 6400,
            'waveforms': {
                'I': 'readout_fast_wf_res_1',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'cos': 'res_1_iw_fast_cosine',
                'sin': 'res_1_iw_fast_sine',
            }
        },
        'readout_16us_pulse_res_2': {
            'operation': 'measurement',
            'length': 16000,
            'waveforms': {
                'I': 'readout_wf_res_2',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'calib_3200ns': 'calib_3200ns_weights',
                'calib_16us': 'calib_16us_weights',
                'I_cos': 'res_2_iw_I_cosine',
                'I_sin': 'res_2_iw_I_sine',
                'J_cos': 'res_2_iw_J_cosine',
                'J_sin': 'res_2_iw_J_sine',
                'K_cos': 'res_2_iw_K_cosine',
                'K_sin': 'res_2_iw_K_sine',
                'L_cos': 'res_2_iw_L_cosine',
                'L_sin': 'res_2_iw_L_sine',
            }
        },
        'readout_4us_pulse_res_2': {
            'operation': 'measurement',
            'length': 4000,
            'waveforms': {
                'I': 'readout_4us_wf_res_2',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'readout_4us': 'readout_4us_weights',
                "cos": "cosine_weights_4us",
                "sin": "sine_weights_4us",
                "minus_sin": "minus_sine_weights_4us",
            }
        },
        'readout_5us_pulse_res_2': {
            'operation': 'measurement',
            'length': 5000,
            'waveforms': {
                'I': 'readout_5us_wf_res_2',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'readout_5us': 'readout_5us_weights',
                "cos": "cosine_weights",
                "sin": "sine_weights",
                "minus_sin": "minus_sine_weights",
            }
        },
        'readout_calib_pulse_res_2': {
            'operation': 'measurement',
            'length': 16000,
            'waveforms': {
                'I': 'readout_wf_res_2',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'calib_3200ns': 'calib_3200ns_weights',
                'calib_16us': 'calib_16us_weights'
            }
        },
        'readout_fast_pulse_res_2': {
            'operation': 'measurement',
            'length': 6400,
            'waveforms': {
                'I': 'readout_fast_wf_res_2',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'cos': 'res_2_iw_fast_cosine',
                'sin': 'res_2_iw_fast_sine',
            }
        },
        'readout_16us_pulse_res_C': {
            'operation': 'measurement',
            'length': 16000,
            'waveforms': {
                'I': 'readout_wf_res_C',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'calib_3200ns': 'calib_3200ns_weights',
                'calib_16us': 'calib_16us_weights',
                'I_cos': 'res_C_iw_I_cosine',
                'I_sin': 'res_C_iw_I_sine',
                'J_cos': 'res_C_iw_J_cosine',
                'J_sin': 'res_C_iw_J_sine',
                'K_cos': 'res_C_iw_K_cosine',
                'K_sin': 'res_C_iw_K_sine',
                'L_cos': 'res_C_iw_L_cosine',
                'L_sin': 'res_C_iw_L_sine',
            }
        },
        'readout_calib_pulse_res_C': {
            'operation': 'measurement',
            'length': 16000,
            'waveforms': {
                'I': 'readout_wf_res_C',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'calib_3200ns': 'calib_3200ns_weights',
                'calib_16us': 'calib_16us_weights'
            }
        },
        'readout_fast_pulse_res_C': {
            'operation': 'measurement',
            'length': 6400,
            'waveforms': {
                'I': 'readout_fast_wf_res_C',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'cos': 'res_C_iw_fast_cosine',
                'sin': 'res_C_iw_fast_sine',
            }
        },
        'readout_4us_pulse_res_C': {
            'operation': 'measurement',
            'length': 4000,
            'waveforms': {
                'I': 'readout_4us_wf_res_C',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'readout_4us': 'readout_4us_weights',
                "cos": "cosine_weights_4us",
                "sin": "sine_weights_4us",
                "minus_sin": "minus_sine_weights_4us",
            }
        },
        'square_pulse': {
            'operation': 'control',
            'length': 16000,
            'waveforms': {
                'I': 'square_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON'
        },
        'square_pad_pulse': {
            'operation': 'control',
            'length': 16016,
            'waveforms': {
                'I': 'square_pad_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON'
        },
        'square_pulse_Q1': {
            'operation': 'control',
            'length': 16000,
            'waveforms': {
                'I': 'square_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON'
        },
        'x180_pulse_Q1': {
            'operation': 'control',
            'length': Q1_DL_x180_len,
            'waveforms': {
                'I': 'x180_wf_Q1',
                'Q': 'zero_wf'
            }
        },
        'x90_pulse_Q1': {
            'operation': 'control',
            'length': Q1_DL_x90_len,
            'waveforms': {
                'I': 'x90_wf_Q1',
                'Q': 'zero_wf'
            }
        },
        'x180_pulse_Q1_DL': {
            'operation': 'control',
            'length': Q1_DL_x180_len,
            'waveforms': {
                'I': 'x180_wf_Q1_DL',
                'Q': 'zero_wf'
            }
        },
        'x360_pulse_Q1_DL': {
            'operation': 'control',
            'length': Q1_DL_x180_len,
            'waveforms': {
                'I': 'x360_wf_Q1_DL',
                'Q': 'zero_wf'
            }
        },
        'x90_pulse_Q1_DL': {
            'operation': 'control',
            'length': Q1_DL_x90_len,
            'waveforms': {
                'I': 'x90_wf_Q1',
                'Q': 'zero_wf'
            }
        },
        'x180_cos_pulse_Q1': {
            'operation': 'control',
            'length': x180_cos_len,
            'waveforms': {
                'I': 'x180_cos_wf_Q1',
                'Q': 'zero_wf'
            }
        },
        'x90_cos_pulse_Q1': {
            'operation': 'control',
            'length': x90_cos_len,
            'waveforms': {
                'I': 'x90_cos_wf_Q1',
                'Q': 'zero_wf'
            }
        },
        'x180_pulse_Q2': {
            'operation': 'control',
            'length': 80,
            'waveforms': {
                'I': 'x180_wf_Q2',
                'Q': 'zero_wf'
            }
        },
        'x90_pulse_Q2': {
            'operation': 'control',
            'length': 40,
            'waveforms': {
                'I': 'x90_wf_Q2',
                'Q': 'zero_wf'
            }
        },
        'x180_pulse_QC': {
            'operation': 'control',
            'length': QC_x180_len,
            'waveforms': {
                'I': 'x180_wf_QC',
                'Q': 'zero_wf'
            }
        },
        'x90_pulse_QC': {
            'operation': 'control',
            'length': QC_x90_len,
            'waveforms': {
                'I': 'x90_wf_QC',
                'Q': 'zero_wf'
            }
        },
        'x180_cos_pulse_QC': {
            'operation': 'control',
            'length': x180_cos_len,
            'waveforms': {
                'I': 'x180_cos_wf_QC',
                'Q': 'zero_wf'
            }
        },
        'x90_cos_pulse_QC': {
            'operation': 'control',
            'length': x90_cos_len,
            'waveforms': {
                'I': 'x90_cos_wf_QC',
                'Q': 'zero_wf'
            }
        },
        'x180_pulse_Q2_DL': {
            'operation': 'control',
            'length': Q2_DL_x180_len,
            'waveforms': {
                'I': 'x180_wf_Q2_DL_I',
                'Q': 'x180_wf_Q2_DL_Q'
            }
        },
        'x90_pulse_Q2_DL': {
            'operation': 'control',
            'length': Q2_DL_x90_len,
            'waveforms': {
                'I': 'x90_wf_Q2_DL_I',
                'Q': 'x90_wf_Q2_DL_Q'
            }
        },
        'x180_cos_pulse_Q1_DL': {
            'operation': 'control',
            'length': x180_cos_len,
            'waveforms': {
                'I': 'x180_cos_wf_Q1_DL',
                'Q': 'zero_wf'
            }
        },
        'x90_cos_pulse_Q1_DL': {
            'operation': 'control',
            'length': x90_cos_len,
            'waveforms': {
                'I': 'x90_cos_wf_Q1_DL',
                'Q': 'zero_wf'
            }
        },
        'x180_drag_pulse_Q1_DL': {
            'operation': 'control',
            'length': Q1_DL_x180_drag_len,
            'waveforms': {
                'I': 'x180_drag_wf_Q1_DL_I',
                'Q': 'x180_drag_wf_Q1_DL_Q'
            }
        },
        'x180_drag_pulse_Q2_DL': {
            'operation': 'control',
            'length': Q2_DL_x180_drag_len,
            'waveforms': {
                'I': 'x180_drag_wf_Q2_DL_I',
                'Q': 'x180_drag_wf_Q2_DL_Q'
            }
        },
        'x180_cos_pulse_Q2_DL': {
            'operation': 'control',
            'length': x180_cos_len,
            'waveforms': {
                'I': 'x180_cos_wf_Q2_DL_I',
                'Q': 'x180_cos_wf_Q2_DL_Q'
            }
        },
        'x90_cos_pulse_Q2_DL': {
            'operation': 'control',
            'length': x90_cos_len,
            'waveforms': {
                'I': 'x90_cos_wf_Q2_DL_I',
                'Q': 'x90_cos_wf_Q2_DL_Q'
            }
        },
        'x180_pulse_Q1_ef_DL': {
            'operation': 'control',
            'length': Q1_ef_DL_x180_len,
            'waveforms': {
                'I': 'x180_wf_Q1_ef_DL',
                'Q': 'zero_wf'
            }
        },
        'x90_pulse_Q1_ef_DL': {
            'operation': 'control',
            'length': Q1_ef_DL_x90_len,
            'waveforms': {
                'I': 'x90_wf_Q1_ef_DL',
                'Q': 'zero_wf'
            }
        },
        'x180_cos_pulse_Q1_ef_DL': {
            'operation': 'control',
            'length': x180_ef_cos_len,
            'waveforms': {
                'I': 'x180_cos_wf_Q1_ef_DL',
                'Q': 'zero_wf'
            }
        },
        'x90_cos_pulse_Q1_ef_DL': {
            'operation': 'control',
            'length': x90_ef_cos_len,
            'waveforms': {
                'I': 'x90_cos_wf_Q1_ef_DL',
                'Q': 'zero_wf'
            }
        },
        'x180_pulse_Q2_ef_DL': {
            'operation': 'control',
            'length': Q2_ef_DL_x180_len,
            'waveforms': {
                'I': 'x180_wf_Q2_ef_DL',
                'Q': 'zero_wf'
            }
        },
        'x90_pulse_Q2_ef_DL': {
            'operation': 'control',
            'length': Q2_ef_DL_x90_len,
            'waveforms': {
                'I': 'x90_wf_Q2_ef_DL',
                'Q': 'zero_wf'
            }
        },
        'x180_cos_pulse_Q2_ef_DL': {
            'operation': 'control',
            'length': x180_ef_cos_len,
            'waveforms': {
                'I': 'x180_cos_wf_Q2_ef_DL',
                'Q': 'zero_wf'
            }
        },
        'x90_cos_pulse_Q2_ef_DL': {
            'operation': 'control',
            'length': x90_ef_cos_len,
            'waveforms': {
                'I': 'x90_cos_wf_Q2_ef_DL',
                'Q': 'zero_wf'
            }
        },
        'x180_pulse_QC_ef': {
            'operation': 'control',
            'length': QC_ef_x180_len,
            'waveforms': {
                'I': 'x180_wf_QC_ef',
                'Q': 'zero_wf'
            }
        },
        'x90_pulse_QC_ef': {
            'operation': 'control',
            'length': QC_ef_x90_len,
            'waveforms': {
                'I': 'x90_wf_QC_ef',
                'Q': 'zero_wf'
            }
        },
        'x180_cos_pulse_QC_ef': {
            'operation': 'control',
            'length': x180_ef_cos_len,
            'waveforms': {
                'I': 'x180_cos_wf_QC_ef',
                'Q': 'zero_wf'
            }
        },
        'x90_cos_pulse_QC_ef': {
            'operation': 'control',
            'length': x90_ef_cos_len,
            'waveforms': {
                'I': 'x90_cos_wf_QC_ef',
                'Q': 'zero_wf'
            }
        },
        'square_pulse_pd_1C': {
            'operation': 'control',
            'length': 500,
            'waveforms': {
                'single': 'square_wf_pd_1C'
            }
        },
        'sqisw_pulse_pd_1C': {
            'operation': 'control',
            'length': 60,
            'waveforms': {
                'single': 'sqisw_wf_pd_1C'
            }
        },
        'square_pulse_pd_12': {
            'operation': 'control',
            'length': 500,
            'waveforms': {
                'single': 'square_wf_pd_12'
            }
        },
        'square_pulse_pd_CZ': {
            'operation': 'control',
            'length': pd_CZ_len,
            'waveforms': {
                'single': 'square_wf_pd_CZ'
            }
        },
        'square_pulse_pd_sqrt_iSWAP': {
            'operation': 'control',
            'length': pd_sqrt_iSWAP_len,
            'waveforms': {
                'single': 'square_wf_pd_sqrt_iSWAP'
            }
        },
        'square_pulse_pd_iSWAP': {
            'operation': 'control',
            'length': pd_iSWAP_len,
            'waveforms': {
                'single': 'square_wf_pd_iSWAP'
            }
        },
        'square_pulse_QC_FFL': {
            'operation': 'control',
            'length': QC_FFL_len,
            'waveforms': {
                'I': 'square_wf_QC_FFL_I',
                'Q': 'square_wf_QC_FFL_Q'
            }
        },
        '125MHz_pulse_Q1_ef': {
            'operation': 'control',
            'length': 16000,
            'digital_marker': '125MHz_wf_Q1_ef'
        },
        '167MHz_pulse_Q1_ef': {
            'operation': 'control',
            'length': 16000,
            'digital_marker': '167MHz_wf_Q1_ef'
        },
        '187.5MHz_pulse_Q1_ef': {
            'operation': 'control',
            'length': 16000,
            'digital_marker': '187.5MHz_wf_Q1_ef'
        },
    },
    'waveforms': {
        'zero_wf': {
            'type': 'constant',
            'sample': 0.0
        },
        'readout_fast_wf_res_1': {
            'type': 'arbitrary',
            'samples': readout_wf_res_1[:6400]
        },
        'readout_fast_wf_res_2': {
            'type': 'arbitrary',
            'samples': readout_wf_res_2[:6400]
        },
        'readout_fast_wf_res_C': {
            'type': 'arbitrary',
            'samples': readout_wf_res_C[:6400]
        },
        'readout_4us_wf_res_1': {
            'type': 'arbitrary',
            'samples': readout_wf_res_1[:4000]
        },
        'readout_5us_wf_res_1': {
            'type': 'arbitrary',
            'samples': readout_wf_res_1[:5000]
        },
        'readout_4us_wf_res_2': {
            'type': 'arbitrary',
            'samples': readout_wf_res_2[:4000]
        },
        'readout_5us_wf_res_2': {
            'type': 'arbitrary',
            'samples': readout_wf_res_2[:5000]
        },
        'readout_wf_res_1': {
            'type': 'arbitrary',
            'samples': readout_wf_res_1
        },
        'readout_wf_res_2': {
            'type': 'arbitrary',
            'samples': readout_wf_res_2
        },
        'readout_wf_res_C': {
            'type': 'arbitrary',
            'samples': readout_wf_res_C
        },
        'readout_4us_wf_res_C': {
            'type': 'arbitrary',
            'samples': readout_wf_res_C[:4000]
        },
        'square_wf': {
            'type': 'constant',
            'sample': 0.4
        },
        'square_pad_wf': {
            'type': 'arbitrary',
            'samples': [0.0] * 16 + [0.4] * 16000
        },
        'x180_wf_Q1': {
            'type': 'constant',
            'sample': Q1_DL_x180_amp
        },
        'x90_wf_Q1': {
            'type': 'constant',
            'sample': Q1_DL_x90_amp
        },
        'x180_cos_wf_Q1': {
            'type': 'arbitrary',
            'samples': Q1_DL_x180_cos_amp * np.real(x180_cos_wf)
        },
        'x90_cos_wf_Q1': {
            'type': 'arbitrary',
            'samples': Q1_DL_x90_cos_amp * np.real(x90_cos_wf)
        },
        'x180_wf_Q2': {
            'type': 'constant',
            'sample': 0.08208
        },
        'x90_wf_Q2': {
            'type': 'constant',
            'sample': 0.08208
        },
        'x180_wf_QC': {
            'type': 'constant',
            'sample': QC_x180_amp
        },
        'x90_wf_QC': {
            'type': 'constant',
            'sample': QC_x90_amp
        },
        'x180_cos_wf_QC': {
            'type': 'arbitrary',
            'samples': QC_x180_cos_amp * np.real(x180_cos_wf)
        },
        'x90_cos_wf_QC': {
            'type': 'arbitrary',
            'samples': QC_x90_cos_amp * np.real(x90_cos_wf)
        },
        'x180_wf_Q1_DL': {
            'type': 'constant',
            'sample': Q1_DL_x180_amp
        },
        'x360_wf_Q1_DL': {
            'type': 'constant',
            'sample': 2*Q1_DL_x180_amp
        },
        'x90_wf_Q1_DL': {
            'type': 'constant',
            'sample': Q1_DL_x90_amp
        },
        'x180_cos_wf_Q1_DL': {
            'type': 'arbitrary',
            'samples': Q1_DL_x180_cos_amp * np.real(x180_cos_wf)
        },
        'x90_cos_wf_Q1_DL': {
            'type': 'arbitrary',
            'samples': Q1_DL_x90_cos_amp * np.real(x90_cos_wf)
        },
        'x180_cos_wf_Q1_DL_I': {
            'type': 'arbitrary',
            'samples': Q1_DL_x180_cos_amp * np.real(x180_cos_wf * np.exp((2j * np.pi) * 0.0 * 1e-3 * (np.arange(x180_cos_len) - 0.5 * (x180_cos_len - 1))))
        },
        'x180_cos_wf_Q1_DL_Q': {
            'type': 'arbitrary',
            'samples': Q1_DL_x180_cos_amp * np.imag(x180_cos_wf * np.exp((2j * np.pi) * 0.0 * 1e-3 * (np.arange(x180_cos_len) - 0.5 * (x180_cos_len - 1))))
        },
        'x90_cos_wf_Q1_DL_I': {
            'type': 'arbitrary',
            'samples': Q1_DL_x90_cos_amp * np.real(x90_cos_wf * np.exp((2j * np.pi) * 0.0 * 1e-3 * (np.arange(x90_cos_len) - 0.5 * (x90_cos_len - 1))))
        },
        'x90_cos_wf_Q1_DL_Q': {
            'type': 'arbitrary',
            'samples': Q1_DL_x90_cos_amp * np.imag(x90_cos_wf * np.exp((2j * np.pi) * 0.0 * 1e-3 * (np.arange(x90_cos_len) - 0.5 * (x90_cos_len - 1))))
        },
        'x180_drag_wf_Q1_DL_I': {
            'type': 'arbitrary',
            'samples': np.asarray(x180_drag_I_Q1, dtype=float).tolist()
        },
        'x180_drag_wf_Q1_DL_Q': {
            'type': 'arbitrary',
            'samples': np.asarray(x180_drag_Q_Q1, dtype=float).tolist()
        },
        'x180_drag_wf_Q2_DL_I': {
            'type': 'arbitrary',
            'samples': np.asarray(x180_drag_I_Q2, dtype=float).tolist()
        },
        'x180_drag_wf_Q2_DL_Q': {
            'type': 'arbitrary',
            'samples': np.asarray(x180_drag_Q_Q2, dtype=float).tolist()
        },
        'x180_wf_Q2_DL': {
            'type': 'constant',
            'sample': Q2_DL_x180_amp
        },
        'x90_wf_Q2_DL': {
            'type': 'constant',
            'sample': Q2_DL_x90_amp
        },
        'x180_wf_Q2_DL_I': {
            'type': 'arbitrary',
            'samples': Q2_DL_x180_amp * np.real(np.exp((2j * np.pi) * 1.3135294516728901 * 1e-3 * (np.arange(Q2_DL_x180_len) - 0.5 * (Q2_DL_x180_len - 1))))
        },
        'x180_wf_Q2_DL_Q': {
            'type': 'arbitrary',
            'samples': Q2_DL_x180_amp * np.imag(np.exp((2j * np.pi) * 1.3135294516728901 * 1e-3 * (np.arange(Q2_DL_x180_len) - 0.5 * (Q2_DL_x180_len - 1))))
        },
        'x90_wf_Q2_DL_I': {
            'type': 'arbitrary',
            'samples': Q2_DL_x90_amp * np.real(np.exp((2j * np.pi) * 1.0030452391713387 * 1e-3 * (np.arange(Q2_DL_x90_len) - 0.5 * (Q2_DL_x90_len - 1))))
        },
        'x90_wf_Q2_DL_Q': {
            'type': 'arbitrary',
            'samples': Q2_DL_x90_amp * np.imag(np.exp((2j * np.pi) * 1.0030452391713387 * 1e-3 * (np.arange(Q2_DL_x90_len) - 0.5 * (Q2_DL_x90_len - 1))))
        },
        'x180_cos_wf_Q2_DL_I': {
            'type': 'arbitrary',
            'samples': Q2_DL_x180_cos_amp * np.real(x180_cos_wf * np.exp((2j * np.pi) * 0.6039780486608042 * 1e-3 * (np.arange(x180_cos_len) - 0.5 * (x180_cos_len - 1))))
        },
        'x180_cos_wf_Q2_DL_Q': {
            'type': 'arbitrary',
            'samples': Q2_DL_x180_cos_amp * np.imag(x180_cos_wf * np.exp((2j * np.pi) * 0.6039780486608042 * 1e-3 * (np.arange(x180_cos_len) - 0.5 * (x180_cos_len - 1))))
        },
        'x90_cos_wf_Q2_DL_I': {
            'type': 'arbitrary',
            'samples': Q2_DL_x90_cos_amp * np.real(x90_cos_wf * np.exp((2j * np.pi) * 0.41517457448854705 * 1e-3 * (np.arange(x90_cos_len) - 0.5 * (x90_cos_len - 1))))
        },
        'x90_cos_wf_Q2_DL_Q': {
            'type': 'arbitrary',
            'samples': Q2_DL_x90_cos_amp * np.imag(x90_cos_wf * np.exp((2j * np.pi) * 0.41517457448854705 * 1e-3 * (np.arange(x90_cos_len) - 0.5 * (x90_cos_len - 1))))
        },
        'x180_cos_wf_Q2_DL': {
            'type': 'arbitrary',
            'samples': Q2_DL_x180_cos_amp * np.real(x180_cos_wf)
        },
        'x90_cos_wf_Q2_DL': {
            'type': 'arbitrary',
            'samples': Q2_DL_x90_cos_amp * np.real(x90_cos_wf)
        },
        'x180_wf_Q1_ef_DL': {
            'type': 'constant',
            'sample': Q1_ef_DL_x180_amp
        },
        'x90_wf_Q1_ef_DL': {
            'type': 'constant',
            'sample': Q1_ef_DL_x90_amp
        },
        'x180_cos_wf_Q1_ef_DL': {
            'type': 'arbitrary',
            'samples': Q1_ef_DL_x180_cos_amp * np.real(x180_ef_cos_wf)
        },
        'x90_cos_wf_Q1_ef_DL': {
            'type': 'arbitrary',
            'samples': Q1_ef_DL_x90_cos_amp * np.real(x90_ef_cos_wf)
        },
        'x180_wf_Q2_ef_DL': {
            'type': 'constant',
            'sample': Q2_ef_DL_x180_amp
        },
        'x90_wf_Q2_ef_DL': {
            'type': 'constant',
            'sample': Q2_ef_DL_x90_amp
        },
        'x180_cos_wf_Q2_ef_DL': {
            'type': 'arbitrary',
            'samples': Q2_ef_DL_x180_cos_amp * np.real(x180_ef_cos_wf)
        },
        'x90_cos_wf_Q2_ef_DL': {
            'type': 'arbitrary',
            'samples': Q2_ef_DL_x90_cos_amp * np.real(x90_ef_cos_wf)
        },
        'x180_wf_QC_ef': {
            'type': 'constant',
            'sample': QC_ef_x180_amp
        },
        'x90_wf_QC_ef': {
            'type': 'constant',
            'sample': QC_ef_x90_amp
        },
        'x180_cos_wf_QC_ef': {
            'type': 'arbitrary',
            'samples': QC_ef_x180_cos_amp * np.real(x180_ef_cos_wf)
        },
        'x90_cos_wf_QC_ef': {
            'type': 'arbitrary',
            'samples': QC_ef_x90_cos_amp * np.real(x90_ef_cos_wf)
        },
        'square_wf_pd_1C': {
            'type': 'constant',
            'sample': 0.5
        },
        'sqisw_wf_pd_1C': {
            'type': 'constant',
            'sample': 0.048
        },
        'square_wf_pd_12': {
            'type': 'constant',
            'sample': 0.5
        },
        'square_wf_pd_CZ': {
            'type': 'constant',
            'sample': 0.5
        },
        'square_wf_pd_sqrt_iSWAP': {
            'type': 'constant',
            'sample': 0.5
        },
        'square_wf_pd_iSWAP': {
            'type': 'constant',
            'sample': 0.5
        },
        'square_wf_QC_FFL': {
            'type': 'constant',
            'sample': 0.5
        },
        'square_wf_QC_FFL_I': {
            'type': 'arbitrary',
            'samples': QC_FFL_amp * np.real(np.exp((2j * np.pi * (QC_FFL_freq - (Q2_freq - Q1_freq) // 2) * 1e-9) * (np.arange(QC_FFL_len) - 0.5 * (QC_FFL_len - 1))))
            # 'samples': QC_FFL_amp * np.real(np.exp((2j * np.pi * 139731 * 1e-9) * (np.arange(QC_FFL_len) - 0.5 * (QC_FFL_len - 1))))
        },
        'square_wf_QC_FFL_Q': {
            'type': 'arbitrary',
            'samples': QC_FFL_amp * np.imag(np.exp((2j * np.pi * (QC_FFL_freq - (Q2_freq - Q1_freq) // 2) * 1e-9) * (np.arange(QC_FFL_len) - 0.5 * (QC_FFL_len - 1))))
            # 'samples': QC_FFL_amp * np.imag(np.exp((2j * np.pi * 139731 * 1e-9) * (np.arange(QC_FFL_len) - 0.5 * (QC_FFL_len - 1))))
        }
    },
    'digital_waveforms': {
        'ON': {
            'samples': [(1, 0)]
        },
        '125MHz_wf_Q1_ef': {
            'samples': [(1, 4), (0, 4)] * 2000
        },
        '167MHz_wf_Q1_ef': {
            'samples': [(1, 3), (0, 3)] * 2666 + [(1, 3), (0, 1)]
        },
        '187.5MHz_wf_Q1_ef': {
            'samples': [(1, 3), (0, 2), (1, 3), (0, 3), (1, 2), (0, 3)] * 1000
        },
        # 'x180_wf_Q1_ef': {
        #     'samples': [(1, 3), (0, 2), (1, 3), (0, 3), (1, 2), (0, 3)] * (Q1_ef_DL_x180_len // 16)
        # },
    },
    'integration_weights': {
        'res_1_iw_I_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_1_iw_I[0], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_1_iw_Q[0], durations)]
        },
        'res_1_iw_I_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_1_iw_Q[0], durations)],
            'sine': [(val, duration) for val, duration in zip(res_1_iw_I[0], durations)]
        },
        'res_1_iw_J_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_1_iw_I[1], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_1_iw_Q[1], durations)]
        },
        'res_1_iw_J_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_1_iw_Q[1], durations)],
            'sine': [(val, duration) for val, duration in zip(res_1_iw_I[1], durations)]
        },
        'res_1_iw_K_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_1_iw_I[2], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_1_iw_Q[2], durations)]
        },
        'res_1_iw_K_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_1_iw_Q[2], durations)],
            'sine': [(val, duration) for val, duration in zip(res_1_iw_I[2], durations)]
        },
        'res_1_iw_L_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_1_iw_I[3], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_1_iw_Q[3], durations)]
        },
        'res_1_iw_L_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_1_iw_Q[3], durations)],
            'sine': [(val, duration) for val, duration in zip(res_1_iw_I[3], durations)]
        },
        'res_2_iw_I_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_2_iw_I[0], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_2_iw_Q[0], durations)]
        },
        'res_2_iw_I_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_2_iw_Q[0], durations)],
            'sine': [(val, duration) for val, duration in zip(res_2_iw_I[0], durations)]
        },
        'res_2_iw_J_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_2_iw_I[1], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_2_iw_Q[1], durations)]
        },
        'res_2_iw_J_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_2_iw_Q[1], durations)],
            'sine': [(val, duration) for val, duration in zip(res_2_iw_I[1], durations)]
        },
        'res_2_iw_K_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_2_iw_I[2], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_2_iw_Q[2], durations)]
        },
        'res_2_iw_K_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_2_iw_Q[2], durations)],
            'sine': [(val, duration) for val, duration in zip(res_2_iw_I[2], durations)]
        },
        'res_2_iw_L_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_2_iw_I[3], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_2_iw_Q[3], durations)]
        },
        'res_2_iw_L_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_2_iw_Q[3], durations)],
            'sine': [(val, duration) for val, duration in zip(res_2_iw_I[3], durations)]
        },
        'res_C_iw_I_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_C_iw_I[0], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_C_iw_Q[0], durations)]
        },
        'res_C_iw_I_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_C_iw_Q[0], durations)],
            'sine': [(val, duration) for val, duration in zip(res_C_iw_I[0], durations)]
        },
        'res_C_iw_J_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_C_iw_I[1], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_C_iw_Q[1], durations)]
        },
        'res_C_iw_J_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_C_iw_Q[1], durations)],
            'sine': [(val, duration) for val, duration in zip(res_C_iw_I[1], durations)]
        },
        'res_C_iw_K_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_C_iw_I[2], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_C_iw_Q[2], durations)]
        },
        'res_C_iw_K_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_C_iw_Q[2], durations)],
            'sine': [(val, duration) for val, duration in zip(res_C_iw_I[2], durations)]
        },
        'res_C_iw_L_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_C_iw_I[3], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_C_iw_Q[3], durations)]
        },
        'res_C_iw_L_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_C_iw_Q[3], durations)],
            'sine': [(val, duration) for val, duration in zip(res_C_iw_I[3], durations)]
        },
        'calib_3200ns_weights': {
            'cosine': [(1.0, 3200)],
            'sine': [(0.0, 3200)]
        },
        'calib_16us_weights': {
            'cosine': [(1.0, 16000)],
            'sine': [(0.0, 16000)]
        },
        'readout_4us_weights': {
            'cosine': [(1.0, 4000)],
            'sine': [(0.0, 4000)]
        },
        'readout_5us_weights': {
            'cosine': [(1.0, 5000)],
            'sine': [(0.0, 5000)]
        },
        "cosine_weights_4us": {
            "cosine": [(1.0, 4000)],
            "sine": [(0.0, 4000)],
        },
        "sine_weights_4us": {
            "cosine": [(0.0, 4000)],
            "sine": [(1.0, 4000)],
        },
        "minus_sine_weights_4us": {
            "cosine": [(0.0, 4000)],
            "sine": [(-1.0, 4000)],
        },
        "cosine_weights": {
            "cosine": [(1.0, 5000)],
            "sine": [(0.0, 5000)],
        },
        "sine_weights": {
            "cosine": [(0.0, 5000)],
            "sine": [(1.0, 5000)],
        },
        "minus_sine_weights": {
            "cosine": [(0.0, 5000)],
            "sine": [(-1.0, 5000)],
        },
        'res_1_iw_fast_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_1_iw_I[0][:18], durations[:18])],
            'sine': [(-val, duration) for val, duration in zip(res_1_iw_Q[0][:18], durations[:18])]
        },
        'res_1_iw_fast_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_1_iw_Q[0][:18], durations[:18])],
            'sine': [(val, duration) for val, duration in zip(res_1_iw_I[0][:18], durations[:18])]
        },
        'res_2_iw_fast_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_2_iw_I[0][:18], durations[:18])],
            'sine': [(-val, duration) for val, duration in zip(res_2_iw_Q[0][:18], durations[:18])]
        },
        'res_2_iw_fast_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_2_iw_Q[0][:18], durations[:18])],
            'sine': [(val, duration) for val, duration in zip(res_2_iw_I[0][:18], durations[:18])]
        },
        'res_C_iw_fast_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_C_iw_I[0][:18], durations[:18])],
            'sine': [(-val, duration) for val, duration in zip(res_C_iw_Q[0][:18], durations[:18])]
        },
        'res_C_iw_fast_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_C_iw_Q[0][:18], durations[:18])],
            'sine': [(val, duration) for val, duration in zip(res_C_iw_I[0][:18], durations[:18])]
        }
    },
    "mixers": {
        "mixer_Q1_DL": [
            {
                "correction": Q1_DL_mixer_C_matrix,
                "intermediate_frequency": if_Q1,
                "lo_frequency": lo_qubit
            }
        ],
        "mixer_Q2_DL": [
            {
                "correction": Q2_DL_mixer_C_matrix,
                "intermediate_frequency": if_Q2,
                "lo_frequency": lo_qubit
            }
        ],
        "mixer_Q1_ef_DL": [
            {
                "correction": Q1_ef_DL_mixer_C_matrix,
                "intermediate_frequency": if_Q1_ef,
                "lo_frequency": lo_qubit
            }
        ],
        "mixer_Q2_ef_DL": [
            {
                "correction": Q2_ef_DL_mixer_C_matrix,
                "intermediate_frequency": if_Q2_ef,
                "lo_frequency": lo_qubit
            }
        ],
        "mixer_readout_line": [
            {
                "correction": QC_mixer_C_matrix,
                "intermediate_frequency": if_QC,
                "lo_frequency": lo_mod
            },
            {
                "correction": QC_ef_mixer_C_matrix,
                "intermediate_frequency": if_QC_ef,
                "lo_frequency": lo_mod
            },
            {
                "correction": res_1_mixer_C_matrix,
                "intermediate_frequency": if_res_1,
                "lo_frequency": lo_res
            },
            {
                "correction": res_2_mixer_C_matrix,
                "intermediate_frequency": if_res_2,
                "lo_frequency": lo_res
            },
            {
                "correction": res_C_mixer_C_matrix,
                "intermediate_frequency": if_res_C,
                "lo_frequency": lo_res
            },
            {
                "correction": ACS_mixer_C_matrix,
                "intermediate_frequency": if_ACS,
                "lo_frequency": lo_qubit
            }
        ],
        "mixer_drive_line1": [
            {
                "correction": [
                    1,
                    0,
                    0,
                    1
                ],
                "intermediate_frequency": if_Q1,
                "lo_frequency": lo_qubit
            },
            {
                "correction": [
                    1,
                    0,
                    0,
                    1
                ],
                "intermediate_frequency": if_Q2,
                "lo_frequency": lo_qubit
            }
        ],
        "mixer_drive_line2": [
            {
                "correction": [
                    1,
                    0,
                    0,
                    1
                ],
                "intermediate_frequency": if_Q2,
                "lo_frequency": lo_qubit
            }
        ],
        "mixer_QC_FFL_dummy": [
            {
                "correction": [
                    1,
                    0,
                    0,
                    0
                ],
                "intermediate_frequency": (Q2_freq - Q1_freq) // 2,
                "lo_frequency": 0
            }
        ]
    }
}
