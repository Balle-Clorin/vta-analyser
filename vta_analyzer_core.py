# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:42:36 2026

@author: ersol
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:08:14 2026

@author: ersol
"""

"""
Vertical Tracking Angle (VTA) & Tracing Distortion Analyzer
============================================================
Software implementation of US Patent 4,359,768 (CBS Inc., 1982)
"Vertical Tracking Angle Meter" — Abbagnaro & Gust
Version 5.1 — Auto-detection of actual tone frequencies (fixes turntable speed error / non-standard cutting frequency)
            — RIAA phase warning: amplitude threshold lowered to 20dB, plus phi-based warning for RIAA records played through RIAA preamp
Version 5.2 — User-configurable reference recording angles RECORDING_ANGLE_VERTICAL / RECORDING_ANGLE_LATERAL
            — Removes hardcoded 16.5°/0.0° from resolve block; STR-111 support via RECORDING_ANGLE_VERTICAL = 15.0
            — Low-F sanity check warns when TONE_PAIR does not match actual record track (F < 5 Hz threshold)

Signal chain (Fig. 4 of patent):

  INPUT (RAW velocity pickup output — NO RIAA equalization applied)
    │
    ├─── SIGNAL PATH A ────────────────────────────────────────────────────
    │    [44] 2kHz HPF + 400Hz NOTCH  →  isolates 4kHz carrier + sidebands
    │    [46] Axis Crossing Detector  →  4kHz carrier → 4kHz square wave
    │    [48] FM Discriminator        →  demodulates 4kHz → 400Hz dev signal
    │    [50] LPF                     →  isolates 400Hz component of FM dev
    │    [52] Phase Compensator       →  corrects HPF residual phase shift
    │         → 400Hz DEVIATION SIGNAL (amplitude ∝ FM deviation F)
    │
    └─── SIGNAL PATH B ────────────────────────────────────────────────────
         [54] LPF                     →  removes 4kHz, passes 400Hz only
         [55] Switch (0° or 90°)      →  selects tracking or tracing mode
         [57] 90° Phase Shift (opt.)  →  inserted for tracing distortion mode
         [56] Axis Crossing Detector  →  400Hz sine → 400Hz square wave
              → 400Hz REFERENCE SQUARE WAVE

  CHOPPER [58]: deviation_signal × reference_square_wave
    → C₀   waveform (full-wave rectified) when signals in phase → TRACKING
    → C₉₀  waveform (zero average)        when in quadrature   → TRACING

  [60] LPF + OFFSET → DC proportional to VTA (offset sets zero at θR=16.5° for CBS)
  [62] METER → reads VTA directly in degrees

VTA Formula (patent Section 3 / Col. 3-4):
    θP  = arctan[ tan(θR) ± (9.1×10⁻⁴  · R · F · cos(φ)) / V₄₀₀ ]  [vertical, any f_low]
θPH = tan⁻¹( 8.72×10⁻⁴ · R · F · cos(φ) / V_SH )                  [lateral,  any f_low]
Coefficients are physical constants — they do NOT scale with f_low.

    θP  = vertical playback angle (degrees)
    θR  = vertical recording angle (16.5° for CBS STR-112 Group 2B — White & Gust measured value)
    R   = groove radius (metres)
    F   = peak 400Hz FM deviation of 4kHz tone (Hz)
    φ   = phase lead of E2 w.r.t. E1 (degrees)  [0° = in-phase]
    V₄₀₀= peak velocity of recorded 400Hz tone (m/s)
    −   = left channel,  + = right channel

CBS STR-112 Test Record:
    Group 2B (vertical modulation, 400+4000 Hz) — VERTICAL TRACKING ANGLE
        Band B1: +6 dB,  Band B2: +9 dB,  Band B3: +12 dB
        Vertical recording angle θR = 16.5°, coefficient = 9.10×10⁻⁴

    Group 1B (lateral modulation, 400+4000 Hz) — LATERAL TRACKING ANGLE
        Bands: +6, +9, +12, +15 dB  (+18 dB not recommended)
        Lateral recording angle θR = 0.0°,  coefficient = 8.73×10⁻⁴
        Set MODULATION = 'lateral' and BAND = '+6dB' (or +9/+12/+15)

Usage:
    python vta_analyzer.py recording.wav [options]
    python vta_analyzer.py --demo          # synthetic signal, no WAV needed

Requirements:
    pip install numpy scipy matplotlib
    pip install soundfile   # for WAV input (optional)
"""

import sys
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as sig

# ── Try soundfile, fall back gracefully ──────────────────────────────────────
try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    _HAS_SF = False

# ═════════════════════════════════════════════════════════════════════════════
#  ███████╗███████╗████████╗████████╗██╗███╗   ██╗ ██████╗ ███████╗
#  ██╔════╝██╔════╝╚══██╔══╝╚══██╔══╝██║████╗  ██║██╔════╝ ██╔════╝
#  ███████╗█████╗     ██║      ██║   ██║██╔██╗ ██║██║  ███╗███████╗
#  ╚════██║██╔══╝     ██║      ██║   ██║██║╚██╗██║██║   ██║╚════██║
#  ███████║███████╗   ██║      ██║   ██║██║ ╚████║╚██████╔╝███████║
#  ╚══════╝╚══════╝   ╚═╝      ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝
#
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │              USER SETTINGS — EDIT HERE BEFORE RUNNING              │
#  │                     Press F5 in Spyder to run                      │
#  └─────────────────────────────────────────────────────────────────────┘
# ═════════════════════════════════════════════════════════════════════════════

# Set DEMO_MODE = True to run without a WAV file (generates a synthetic signal)
# Set DEMO_MODE = False to analyse your own recording
DEMO_MODE       = False

# ── Your WAV file (only used when DEMO_MODE = False) ─────────────────────────
# Windows:   r"C:\Users\YourName\recordings\track2.wav"
# Mac/Linux: "/home/yourname/recordings/track2.wav"
AUDIO_FILE      = r"2026 GYRO OC9 CBS STR-112 imd vertical 6db 400-4000.wav"

# ── Calibration mode ─────────────────────────────────────────────────────────
# Set CALIBRATION_MODE = True to calculate the correct PEAK_VEL for your
# recording setup. Two methods are run and cross-checked:
#   Method A: WAV amplitude ratio × carrier velocity from label (linear chain)
#   Method B: Known cartridge VTA back-calculation (corrects non-linearity)
# Set KNOWN_VTA = None to run Method A only.
# Set KNOWN_VTA to your cartridge's true VTA to run both methods.
CALIBRATION_MODE    = False
KNOWN_VTA           = 23.0    # degrees — your cartridge's true VTA (Method B only)

# ── Calibration method ────────────────────────────────────────────────────────
# 'A'    — WAV amplitude ratio × label carrier velocity.
#          Requires the carrier displacement stated on the record label.
#          Recommended for CBS STR-112 (label values known).
#          Set REF_DISPLACEMENT_UM and REF_CARRIER_DB from your record label.
#
# 'B'    — Back-calculates PEAK_VEL from a known cartridge VTA.
#          Works for ANY record. Requires KNOWN_VTA to be set.
#          Recommended for Analogue Magic and all custom records.
#
# 'both' — Runs both methods and reports agreement between them.
#          Only valid for CBS STR-112 where label values are known.
CALIBRATION_METHOD  = 'both'

# Method A label reference values (CBS STR-112 only):
REF_DISPLACEMENT_UM = 11.2    # µm  — carrier reference displacement from label
REF_CARRIER_DB      = -18.0   # dB  — carrier level relative to REF_DISPLACEMENT_UM   # dB — carrier level relative to reference
# The patent (Col 7) notes that L/R channel wiring affects the sign convention.
# If your VTA results look wrong (too low, near 0° or negative) swap this.
# How to check: run the script, if L channel reads ~6° and R reads ~24°
# (or vice versa) set SWAP_CHANNELS = True to correct it.
# This is equivalent to the front-panel channel switch on the hardware meter.
SWAP_CHANNELS   = False

# ── Which stereo channel(s) to analyse ───────────────────────────────────────
CHANNEL         = 'both'   # 'L', 'R', or 'both' (recommended)

# ══════════════════════════════════════════════════════════════════════════════
#  RECORD MODE — choose ONE of the two sections below
# ══════════════════════════════════════════════════════════════════════════════

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  MODE 1 — CBS STR-112  (auto-configured from tables)                    │
# │  Set the three parameters below. All other values are set automatically. │
# └──────────────────────────────────────────────────────────────────────────┘

# Tone pair:   '400+4000' = outer half Side B  |  '200+4000' = inner half Side B or custom
TONE_PAIR       = '400+4000'

# Modulation:  'vertical' = VTA measurement (Group 2B)
#              'lateral'  = HTA measurement (Group 1B)
MODULATION      = 'vertical'

# Band level:  '+6dB' | '+9dB' | '+12dB'            (all tone pairs, both modes)
#              '+15dB'                                (lateral only)
#   (+18dB not included — distortion too high for reliable tracking)
#
#   Full reference (PEAK_VEL and GROOVE_RADIUS set automatically):
#
#   TONE_PAIR    MODULATION   BAND    PEAK_VEL  GROOVE_RADIUS
#   400+4000     vertical     +6dB    0.0563    0.114   ← White & Gust Table 2, θR=16.5°
#   400+4000     vertical     +9dB    0.0796    0.110
#   400+4000     vertical     +12dB   0.1126    0.105
#   400+4000     lateral      +6dB    0.0563    0.138
#   400+4000     lateral      +9dB    0.0796    0.135
#   400+4000     lateral      +12dB   0.1126    0.132
#   400+4000     lateral      +15dB   0.1592    0.128
#   200+4000     vertical     +6dB    0.02815   0.077   ← constant-displacement
#   200+4000     vertical     +9dB    0.03980   0.072
#   200+4000     vertical     +12dB   0.05630   0.067
#   200+4000     lateral      +6dB    0.02815   0.100
#   200+4000     lateral      +9dB    0.03980   0.097
#   200+4000     lateral      +12dB   0.05630   0.094
#   200+4000     lateral      +15dB   0.07960   0.089
BAND            = '+6dB'

# ── Reference recording angles (θR) ──────────────────────────────────────────
# These are the angles at which the test record was cut.
# They are used as the reference point (zero-error) in the VTA/HTA formula.
#
# VERTICAL (θR for VTA measurement):
#   16.5° — CBS STR-112, measured by White & Gust (1979) using 11 pickups.
#            This is the most reliably calibrated value available.
#   15.0° — Early records (pre-1970s IEC standard), e.g. CBS STR-111 (1966).
#            Springback correction was not applied when the STR-111 was cut,
#            so the effective groove angle may differ from the cutter setting.
#            Treat STR-111 VTA results as relative measurements only.
#   20.0° — Some later test records and direct-cut audiophile pressings.
#   Set to None to use the default: 16.5° for CBS STR-112, 0° for lateral.
#
# LATERAL (θR for HTA measurement):
#   0.0°  — Correct for all standard stereo records (zero lateral offset).
#            Only change this if your test record was deliberately cut with
#            a known lateral offset angle (uncommon).
RECORDING_ANGLE_VERTICAL = 16.5   # degrees — θR for VTA (vertical modulation)
RECORDING_ANGLE_LATERAL  = 0.0    # degrees — θR for HTA (lateral modulation)

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  MODE 2 — CUSTOM  (any other test record, or manual CBS override)        │
# │                                                                          │
# │  Set TONE_PAIR = 'custom' above, then edit the five values below.        │
# │  These values are always present but only used when TONE_PAIR = 'custom'.│
# └──────────────────────────────────────────────────────────────────────────┘
CUSTOM_MODULATION      = 'vertical'  # 'vertical' or 'lateral'
CUSTOM_F_MOD           = 399       # low modulating frequency (Hz) e.g. 60, 200, 400
CUSTOM_F_CARRIER       = 4022      # carrier frequency (Hz) e.g. 4000, 7000
CUSTOM_PEAK_VEL        = 0.092      # peak source velocity (m/s) from record spec
CUSTOM_GROOVE_RADIUS   = 0.122       # groove radius (m) — measure on record with ruler
CUSTOM_RECORDING_ANGLE = 23       # recording angle (°): use RECORDING_ANGLE_VERTICAL /
                                  # RECORDING_ANGLE_LATERAL above for MODE 1 (CBS).
                                  # For MODE 2 (custom), set this value directly.
                                  # 16.5° = CBS STR-112 vertical (White & Gust)
                                  # 0°    = standard lateral

# ── Inverse RIAA correction ──────────────────────────────────────────────────
# Set to True if you recorded a FLAT test record (e.g. CBS STR-112) through
# a RIAA phono stage and cannot re-record with flat/bypass.
# The script will apply an inverse RIAA curve to the WAV before analysis,
# restoring the correct amplitude relationship between the two tones.
#
# Leave False for:
#   - Flat records recorded with flat/bypass preamp  (normal CBS STR-112 use)
#   - RIAA records (e.g. Analogue Magic) recorded with RIAA playback applied
#     (the RIAA curve is intentional and already accounted for in PEAK_VEL)
# APPLY_INVERSE_RIAA removed — not needed, see comment above

# ── Turntable speed (RPM) ─────────────────────────────────────────────────────
# Used to compute groove tangential velocity c = 2π × R × rpm/60.
# This is needed to correctly scale the formula coefficient for each groove
# radius — the coefficient 9.1e-4 in White & Gust was derived at R=0.114m
# (33⅓ rpm), and varies slightly across bands as R changes.
# Standard: 33.333 for 33⅓ rpm,  45.0 for 45 rpm
TURNTABLE_RPM   = 33.333

# ── Plots to show ─────────────────────────────────────────────────────────────
SHOW_SPECTRUM     = False
SHOW_SIGNAL_CHAIN = False
SHOW_METER        = True

# ── Meter description ─────────────────────────────────────────────────────────
METER_DESCRIPTION = "2026 GYRO OC9 CBS STR-112 SIDE B IMD 6db 400:4k vertical"
# ── Demo mode only ────────────────────────────────────────────────────────────
DEMO_VTA_ERROR  = 6.0   # degrees — VTA error injected in demo signal
#                          6° with imd_factor=3 gives ~6% IM% matching real STR-111 data

# ═════════════════════════════════════════════════════════════════════════════
#  END OF USER SETTINGS  —  do not edit below this line
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
#  LATERAL BAND LOOKUP TABLE  (White & Gust Table 2, Group 1B)
# ═════════════════════════════════════════════════════════════════════════════

# Band tables — White & Gust Table 2 (400Hz) and record measurements (200Hz)
# PEAK_VEL = V_sv / V_SH in m/s.  Groove radii in metres.
# 200Hz and 400Hz bands share the same groove velocities (constant velocity
# recording) but are at different radii (inner vs outer half of Side B).

_VERTICAL_BANDS = {
    # 400Hz + 4kHz  (outer half Side B, Group 2B)
    '400+4000': {
        '+6dB':  {'peak_vel': 0.0563, 'groove_radius': 0.114},
        '+9dB':  {'peak_vel': 0.0796, 'groove_radius': 0.110},
        '+12dB': {'peak_vel': 0.1126, 'groove_radius': 0.105},
    },
    # 200Hz + 4kHz  (inner half Side B)
    # PEAK_VEL = V_sv(400Hz) × (200/400): constant-displacement recording.
    # Theoretical estimate — verify with CALIBRATION MODE on your pressing.
    '200+4000': {
        '+6dB':  {'peak_vel': 0.02815, 'groove_radius': 0.077},
        '+9dB':  {'peak_vel': 0.03980, 'groove_radius': 0.072},
        '+12dB': {'peak_vel': 0.05630, 'groove_radius': 0.067},
    },
}


_BANDS = {
    # 400Hz + 4kHz  (outer half Side B, Group 1B)
    '400+4000': {
        '+6dB':  {'peak_vel': 0.0563, 'groove_radius': 0.138},
        '+9dB':  {'peak_vel': 0.0796, 'groove_radius': 0.135},
        '+12dB': {'peak_vel': 0.1126, 'groove_radius': 0.132},
        '+15dB': {'peak_vel': 0.1592, 'groove_radius': 0.128},
    },
    # 200Hz + 4kHz  (inner half Side B)
    # PEAK_VEL = V_sv(400Hz) × (200/400): constant-displacement recording.
    # Theoretical estimate — verify with CALIBRATION MODE on your pressing.
    '200+4000': {
        '+6dB':  {'peak_vel': 0.02815, 'groove_radius': 0.100},
        '+9dB':  {'peak_vel': 0.03980, 'groove_radius': 0.097},
        '+12dB': {'peak_vel': 0.05630, 'groove_radius': 0.094},
        '+15dB': {'peak_vel': 0.07960, 'groove_radius': 0.089},
    },
}


# ── Resolve settings from MODE 1 (CBS STR-112) or MODE 2 (custom) ────────────
if TONE_PAIR == 'custom':
    # MODE 2: apply CUSTOM_ values directly
    MODULATION      = CUSTOM_MODULATION
    F_MOD           = CUSTOM_F_MOD
    F_CARRIER       = CUSTOM_F_CARRIER
    PEAK_VEL        = CUSTOM_PEAK_VEL
    GROOVE_RADIUS   = CUSTOM_GROOVE_RADIUS
    RECORDING_ANGLE = CUSTOM_RECORDING_ANGLE
else:
    # MODE 1: CBS STR-112 — auto-configure from lookup tables
    if TONE_PAIR == '200+4000':
        F_MOD     = 200
        F_CARRIER = 4000
    else:   # '400+4000'
        F_MOD     = 400
        F_CARRIER = 4000

    _band_table = (_BANDS if MODULATION == 'lateral'
                   else _VERTICAL_BANDS)
    _tone_table = _band_table.get(TONE_PAIR, {})
    if BAND not in _tone_table:
        _valid = list(_tone_table.keys())
        raise ValueError(
            f"BAND='{BAND}' not valid for "
            f"MODULATION='{MODULATION}', TONE_PAIR='{TONE_PAIR}'. "
            f"Valid choices: {_valid}")
    PEAK_VEL      = _tone_table[BAND]['peak_vel']
    GROOVE_RADIUS = _tone_table[BAND]['groove_radius']
    # Use user-configurable recording angles (set above).
    # RECORDING_ANGLE_VERTICAL defaults to 16.5° (White & Gust CBS STR-112).
    # RECORDING_ANGLE_LATERAL  defaults to 0.0°  (standard stereo geometry).
    RECORDING_ANGLE = (RECORDING_ANGLE_LATERAL  if MODULATION == 'lateral'
                       else RECORDING_ANGLE_VERTICAL)

# ═════════════════════════════════════════════════════════════════════════════
#  RECORD / INSTRUMENT CONSTANTS  (adjust to match your setup)
# ═════════════════════════════════════════════════════════════════════════════

# Recording angle is set in USER SETTINGS above as RECORDING_ANGLE — do not redefine here
# F_LOW and F_HIGH are set in the USER SETTINGS block at the top of this file
# as F_MOD and F_CARRIER — do not add them here
# GROOVE_RADIUS is set in USER SETTINGS above — do not redefine here.
# For lateral bands it is overridden automatically by the BAND lookup.
# Peak velocity is set in USER SETTINGS above as PEAK_VEL — do not redefine here.
# Physics-derived value: V_sv = 0.0563 m/s for +6dB vertical (White & Gust Table 2)
# This is a physical constant of the record, not dependent on the cartridge.

# ═════════════════════════════════════════════════════════════════════════════
#  UTILITY: FILTERS
# ═════════════════════════════════════════════════════════════════════════════

def _sos_bp(fs, fc, bw, order=4):
    lo = max(fc - bw/2, 1.0)
    hi = min(fc + bw/2, fs/2 - 1)
    return sig.butter(order, [lo, hi], btype='bandpass', fs=fs, output='sos')

def _sos_hp(fs, fc, order=4):
    return sig.butter(order, min(fc, fs/2-1), btype='high', fs=fs, output='sos')

def _sos_lp(fs, fc, order=4):
    return sig.butter(order, min(fc, fs/2-1), btype='low',  fs=fs, output='sos')

def _sos_notch(fs, fc, Q=30):
    b, a = sig.iirnotch(fc, Q, fs)
    return sig.tf2sos(b, a)

def apply(sos, x):
    return sig.sosfiltfilt(sos, x)


# ═════════════════════════════════════════════════════════════════════════════
#  INVERSE RIAA FILTER
# ═════════════════════════════════════════════════════════════════════════════

# NOTE: Functions below are kept for reference only — no longer called.
# The FM discriminator method is inherently RIAA-immune (verified).
def _make_inverse_riaa_sos(fs: float):
    """
    Design a digital inverse RIAA filter (IEC 60098 time constants).

    Used when a flat test record (CBS STR-112) was accidentally recorded
    through a RIAA phono stage. Applying this restores the correct flat
    amplitude relationship between the two tones before analysis.

    RIAA time constants (IEC 60098):
      t1 = 3180 µs  →  f1 =   50 Hz  (pole of RIAA playback)
      t2 =  318 µs  →  f2 =  500 Hz  (zero of RIAA playback)
      t3 =   75 µs  →  f3 = 2122 Hz  (pole of RIAA playback)

    Inverse RIAA zeros at f1 and f3, pole at f2.
    Extra stabilising pole at 90kHz makes the filter proper.
    Implemented via bilinear_zpk for maximum accuracy.

    Ratio error between tone pairs (what affects VTA measurement):
      CBS  400+4000 Hz:  -0.15 dB  →  ~0.02° VTA error  (negligible)
      CBS  200+4000 Hz:  -0.15 dB  →  ~0.02° VTA error  (negligible)
      AM    60+7000 Hz:  -0.56 dB  →  ~0.06° VTA error  (negligible)
    All well within the ±5% measurement tolerance of the FM method.

    NOT needed for:
      - Flat records played back flat/bypass        (normal CBS use)
      - RIAA records played back with RIAA applied  (Analogue Magic etc.)
    """
    from scipy.signal import bilinear_zpk, zpk2sos
    t1, t2, t3 = 3180e-6, 318e-6, 75e-6

    # s-domain: zeros at -1/t1 and -1/t3, pole at -1/t2
    # Extra pole at -2π×90kHz stabilises the improper transfer function
    z_s = np.array([-1/t1, -1/t3])
    p_s = np.array([-1/t2, -2*np.pi*90000])

    # Gain normalised to 0 dB at 1 kHz
    w1k = 2*np.pi*1000
    k_s = abs(np.prod(1j*w1k - p_s)) / abs(np.prod(1j*w1k - z_s))

    z_d, p_d, k_d = bilinear_zpk(z_s, p_s, k_s, fs=fs)
    return zpk2sos(z_d, p_d, k_d)


def apply_inverse_riaa(audio: np.ndarray, fs: float) -> np.ndarray:
    """Apply inverse RIAA correction to a stereo or mono audio array."""
    from scipy.signal import sosfiltfilt
    sos = _make_inverse_riaa_sos(fs)
    if audio.ndim == 1:
        return sosfiltfilt(sos, audio)
    return np.column_stack([sosfiltfilt(sos, audio[:, ch])
                             for ch in range(audio.shape[1])])


# ═════════════════════════════════════════════════════════════════════════════
#  SIGNAL PATH A  — FM discriminator chain
# ═════════════════════════════════════════════════════════════════════════════

def path_a(audio: np.ndarray, fs: float, f_low=F_MOD, f_high=F_CARRIER) -> np.ndarray:
    """
    Implements blocks [44]→[46]→[48]→[50]→[52] of Fig. 4.

    [44] 2kHz HPF + 400Hz notch → carrier band only
    [46] Axis crossing detector → 4kHz square wave  (implicit in Hilbert FM)
    [48] FM discriminator       → instantaneous frequency deviation
    [50] LPF                    → isolate 400Hz deviation component
    [52] Phase compensator      → correct HPF residual phase (auto-calibrated)

    Returns: 400Hz deviation signal (amplitude ∝ F, the peak FM deviation Hz)
    """
    # [44] 2kHz HPF + 400Hz notch  (let sidebands around 4kHz pass)
    hpf      = apply(_sos_hp(fs, 2000.0, order=4), audio)
    notch    = apply(_sos_notch(fs, f_low, Q=20),  hpf)

    # [46]+[48] Zero-crossing FM discriminator via instantaneous frequency
    #   The axis-crossing detector converts the carrier to a square wave;
    #   this is equivalent to taking the sign() then differentiating — in
    #   DSP the Hilbert-transform approach gives the same instantaneous freq.
    analytic  = sig.hilbert(notch)
    inst_ph   = np.unwrap(np.angle(analytic))
    inst_freq = np.concatenate([[0], np.diff(inst_ph)]) * fs / (2*np.pi)
    deviation = inst_freq - f_high          # FM deviation in Hz (baseband)

    # [50] LPF — isolate f_low component of FM deviation
    dev_filt  = apply(_sos_lp(fs, f_low * 2.5, order=4), deviation)

    # Note: block [52] phase compensator (hardware: RC network to correct HPF
    # residual phase at f_high) is NOT needed in DSP — the Hilbert-transform
    # FM discriminator has no residual phase error at the modulating frequency.
    # Applying software phase rotation here would destroy the phase information
    # that the chopper needs to separate tracking from tracing components.

    return dev_filt                         # units: Hz of FM deviation


def _phase_compensate(dev: np.ndarray, fs: float, f_low: float) -> np.ndarray:
    """
    Block [52]: correct residual phase introduced by the HPF section of [44].
    Implemented as a fractional-sample circular shift via FFT phase rotation.
    """
    t       = np.arange(len(dev)) / fs
    ref     = np.sin(2*np.pi*f_low*t)
    # find phase of f_low component
    cc      = np.dot(dev, ref) / len(dev)
    ss      = np.dot(dev, np.cos(2*np.pi*f_low*t)) / len(dev)
    phi_rad = np.arctan2(cc, ss)          # phase error

    # rotate in frequency domain
    D = np.fft.rfft(dev)
    freqs = np.fft.rfftfreq(len(dev), 1/fs)
    # apply correction only near f_low; elsewhere leave unchanged
    # (hardware phase compensator is wideband — approximate that here)
    correction = np.exp(-1j * phi_rad * np.exp(-((freqs - f_low)**2) / (f_low**2)))
    return np.fft.irfft(D * correction, n=len(dev))


# ═════════════════════════════════════════════════════════════════════════════
#  SIGNAL PATH B  — 400Hz reference square wave
# ═════════════════════════════════════════════════════════════════════════════

def path_b(audio: np.ndarray, fs: float,
           f_low=F_MOD, f_high=F_CARRIER,
           mode='tracking') -> np.ndarray:
    """
    Implements blocks [54]→[55/57]→[56] of Fig. 4.

    [54] LPF         → pass 400Hz, remove 4kHz
    [55] Mode switch → 0° (tracking) or 90° (tracing)
    [57] 90° shifter → inserted in tracing mode
    [56] Axis crossing detector → 400Hz square wave

    mode='tracking' → square wave in phase with 400Hz record signal
                       → chopper sees REAL component → VTA reading
    mode='tracing'  → square wave shifted 90° → sees QUADRATURE component
                       → tracing distortion reading
    """
    # [54] LPF: remove carrier
    ref_sine  = apply(_sos_lp(fs, f_low * 3, order=4), audio)
    # narrow bandpass to clean up the 400Hz tone
    ref_sine  = apply(_sos_bp(fs, f_low, f_low*0.8, order=4), ref_sine)

    if mode == 'tracing':
        # [57] 90° phase shift — implemented via Hilbert transform
        analytic = sig.hilbert(ref_sine)
        ref_sine = np.imag(analytic)     # 90° shifted

    # [56] Axis crossing detector: sine → square wave
    square = np.sign(ref_sine)
    square[square == 0] = 1.0            # avoid zeros at crossings
    return square


# ═════════════════════════════════════════════════════════════════════════════
#  CHOPPER  [58]  +  LPF  [60]
# ═════════════════════════════════════════════════════════════════════════════

def chopper_lpf(deviation: np.ndarray, square: np.ndarray,
                fs: float, f_low=F_MOD) -> tuple:
    """
    Block [58]: multiply deviation signal by reference square wave.

    When in-phase (tracking): output ≈ full-wave rectified sine → positive DC
    When quadrature (tracing): output is odd-symmetric → zero DC average

    Block [60]: LPF to extract DC proportional to angular mismatch.

    Returns: (dc_value, chopped_waveform)
    """
    chopped = deviation * square

    # [60] LPF — cutoff well below f_low to average over many cycles
    # LPF cutoff at f_low/40: averages over ~40 cycles of the modulating
    # tone. This rejects wow (0.5-3 Hz beat products) and flutter beat
    # products while preserving the DC tracking/tracing component.
    dc_sig  = apply(_sos_lp(fs, f_low / 40, order=4), chopped)
    dc_val  = float(np.mean(dc_sig[len(dc_sig)//4:]))  # skip filter transient

    return dc_val, chopped, dc_sig


# ═════════════════════════════════════════════════════════════════════════════
#  VTA COMPUTATION  (patent formula, Col. 3-4)
# ═════════════════════════════════════════════════════════════════════════════

def compute_vta(F_signed_hz: float,
                R: float       = GROOVE_RADIUS,
                V_low: float   = PEAK_VEL,
                theta_r: float = RECORDING_ANGLE,
                f_low: float   = F_MOD,
                f_high: float  = F_CARRIER,
                phi_deg: float = 0.0,
                modulation: str = MODULATION) -> float:
    """
    θP = arctan[ tan(θR) + C · R · F_signed / V_low ]

    Matches White & Gust (1979) Table 1 and patent US 4,359,768 Col. 3-4, where:
      C     = 9.1e-4   vertical  (physical constant, independent of f_low)
            = 8.72e-4  lateral   (physical constant, independent of f_low)
      V_low = V_sv or V_SH      [groove source velocity in m/s]
      F_signed = signed FM deviation  [sign from chopper DC × channel polarity]
      theta_r  = 16.5° CBS vertical (White & Gust measured value), 0° lateral
    Valid for both 400Hz and 200Hz modulating frequencies.

    NOTE on cos(φ):
      The paper formula uses F·cos(φ) to extract the real (tracking) component.
      In this implementation the chopper [58] already performs this extraction
      implicitly: multiplying the deviation signal by the reference square wave
      and low-pass filtering yields a DC proportional to F·cos(φ). The sign of
      this DC, combined with channel polarity, gives F_signed directly.
      Therefore cos(φ) must NOT be applied again here — it would double-correct
      and produce wrong results, especially on the left channel where the
      polarity inversion shifts the apparent phi by ~180°.

    PEAK_VEL is user-adjustable. Its physically correct value for CBS STR-112
    is V_sv (vertical) or V_SH (lateral) directly from White & Gust Table 2
    (e.g. 0.0563 m/s for +6dB). Works for any cartridge with flat response.
    """
    theta_R = np.radians(theta_r)
    # Coefficient derived from first principles (White & Gust appendix A5):
    #
    #   C = 2π × rpm / (60 × f_carrier × cos(θR))
    #
    # where f_carrier is the HIGH frequency tone (4kHz, 7kHz, etc.)
    # and θR is the recording angle (16° vertical, 0° lateral).
    # R cancels in the derivation so C is independent of groove radius.
    # C depends only on turntable RPM, carrier frequency, and recording angle.
    #
    # This formula gives the correct coefficient for ANY test record:
    #   CBS STR-112  4kHz carrier  33⅓rpm  16°: C = 9.08e-4  ≈ 9.1e-4 ✓
    #   CBS STR-112  4kHz carrier  33⅓rpm   0°: C = 8.73e-4  ≈ 8.72e-4 ✓
    #   Analogue Magic 7kHz carrier 33⅓rpm 15°: C = 5.16e-4  (different!)
    #
    # The lateral coefficient (8.72e-4) is simply C with θR=0°:
    #   cos(0°) = 1.0  vs  cos(16°) = 0.9613
    # giving ratio 0.9613 ≈ 8.72e-4/9.1e-4 — no separate factor needed.
    # C = 2π × rpm / (60 × f_carrier × cos(θR))
    # θR = 16° vertical, 0° lateral — the cos(θR) difference between
    # vertical (9.1e-4) and lateral (8.72e-4) comes purely from this.
    # No separate geometric factor is needed.
    C = 2.0 * np.pi * TURNTABLE_RPM / (
        60.0 * f_high * np.cos(np.radians(theta_r)))
    factor  = C * R * F_signed_hz / V_low
    return float(np.degrees(np.arctan(np.tan(theta_R) + factor)))


# ═════════════════════════════════════════════════════════════════════════════
#  PEAK FM DEVIATION  from DC output of chopper chain
# ═════════════════════════════════════════════════════════════════════════════

def extract_F_and_phi(dev_signal: np.ndarray, audio: np.ndarray,
                      fs: float, f_low=F_MOD) -> tuple:
    """
    Extract peak FM deviation F (Hz) and phase φ (degrees).

    F is recovered from the RMS of the 400Hz band of the deviation signal.
    φ is the phase lead of E2 (deviation) w.r.t. E1 (400Hz on record).

    In the patent this is done implicitly by the chopper, but we also
    compute F and φ explicitly so we can apply the VTA formula.
    """
    t   = np.arange(len(dev_signal)) / fs

    # Narrow bandpass around f_low in deviation signal → E2
    e2  = apply(_sos_bp(fs, f_low, f_low*0.6, order=4), dev_signal)

    # E1: 400Hz from record (path B before axis-crossing)
    e1_raw = apply(_sos_lp(fs, f_low*3, order=4), audio)
    e1  = apply(_sos_bp(fs, f_low, f_low*0.6, order=4), e1_raw)

    # Peak FM deviation and phase — computed per segment then averaged.
    # Segmenting over 0.5s windows means:
    #   - Each segment spans 200 cycles of 400Hz — plenty for stable statistics
    #   - Wow (0.5-3 Hz) varies slowly across segments; median rejects outliers
    #   - Flutter residuals that slip through the BPF are averaged down
    #   - Any single noisy segment (e.g. a tick or pop) is rejected by median
    seg_len = max(int(fs * 0.5), int(fs / f_low) * 20)  # at least 20 cycles
    n_segs  = max(1, len(e2) // seg_len)

    F_segs    = []
    phi_segs  = []

    for i in range(n_segs):
        sl   = slice(i * seg_len, (i + 1) * seg_len)
        e2s  = e2[sl]
        e1s  = e1[sl]
        ts   = t[sl]

        # F_peak for this segment
        F_segs.append(np.sqrt(2.0) * float(np.std(e2s)))

        # Phase for this segment via phasor correlation
        rc = np.cos(2*np.pi*f_low*ts)
        rs = np.sin(2*np.pi*f_low*ts)
        ph_e1 = np.angle(complex(np.dot(e1s, rc), np.dot(e1s, rs)))
        ph_e2 = np.angle(complex(np.dot(e2s, rc), np.dot(e2s, rs)))
        phi_segs.append(float(np.degrees(ph_e2 - ph_e1)))

    # Median across segments — robust against wow-induced outliers and clicks
    F_peak  = float(np.median(F_segs))

    # Circular median for phase (handles ±180° wrap correctly)
    phi_rad_segs = np.radians(phi_segs)
    phi_deg = float(np.degrees(np.arctan2(
        np.median(np.sin(phi_rad_segs)),
        np.median(np.cos(phi_rad_segs))
    )))

    # Normalise to (−180, +180]
    phi_deg = (phi_deg + 180) % 360 - 180

    return F_peak, phi_deg


# ═════════════════════════════════════════════════════════════════════════════
#  TRACING DISTORTION MEASUREMENT
# ═════════════════════════════════════════════════════════════════════════════

def tracing_distortion(dev_signal: np.ndarray, fs: float,
                       f_low=F_MOD, n_harm=6) -> dict:
    """
    Measure the amplitude of FM deviation harmonics at n·f_low.
    These arise from stylus tracing distortion (quadrature component).
    Returns {n: dB_relative_to_fundamental}
    """
    N      = len(dev_signal)
    freqs  = np.fft.rfftfreq(N, 1/fs)
    spec   = np.abs(np.fft.rfft(dev_signal * np.hanning(N))) * 2 / N

    def peak_bin(fc, bw=20.0):
        mask = (freqs >= fc-bw) & (freqs <= fc+bw)
        return float(np.max(spec[mask])) if np.any(mask) else 0.0

    fund = peak_bin(f_low)
    out  = {}
    for n in range(1, n_harm+1):
        a = peak_bin(f_low * n)
        out[n] = 20*np.log10(max(a, 1e-12) / max(fund, 1e-12))
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  ACTUAL TONE FREQUENCY DETECTION
#  Finds the true frequency of each tone in the recording via parabolic
#  interpolation on the FFT magnitude peak.  This corrects for:
#    • Turntable speed error  (e.g. 33⅓ rpm running slow/fast)
#    • Records cut at non-standard frequencies (e.g. 399 Hz instead of 400 Hz)
#  Without this correction a 1 Hz offset between the nominal and actual f_low
#  causes a 1 Hz beat in the chopper reference, rotating φ through 360° once
#  per second — making phase (and therefore HTA/VTA) essentially random.
# ═════════════════════════════════════════════════════════════════════════════

def detect_actual_frequency(audio: np.ndarray, fs: float,
                             f_nominal: float,
                             search_range: float = None) -> float:
    """
    Return the actual frequency of the dominant tone near f_nominal (Hz).

    Uses a long-FFT magnitude spectrum for frequency resolution, then
    parabolic interpolation between the three bins around the peak for
    sub-bin precision.  Works for both the low modulating tone and the
    high carrier tone.

    Parameters
    ----------
    audio        : input audio (one channel, float array)
    fs           : sample rate (Hz)
    f_nominal    : expected frequency (Hz) — centre of search window
    search_range : half-width of search window (Hz).
                   Defaults to 5 % of f_nominal, which covers ±3 % speed
                   error with plenty of margin.

    Returns
    -------
    Detected frequency in Hz.  Falls back to f_nominal if detection fails.
    """
    if search_range is None:
        search_range = max(f_nominal * 0.05, 5.0)   # at least ±5 Hz

    # Use the full signal for maximum frequency resolution.
    # Hann window suppresses spectral leakage from neighbouring tones.
    N     = len(audio)
    win   = np.hanning(N)
    spec  = np.abs(np.fft.rfft(audio * win))
    freqs = np.fft.rfftfreq(N, 1.0 / fs)

    # Restrict search to window around f_nominal
    mask  = (freqs >= f_nominal - search_range) & \
            (freqs <= f_nominal + search_range)
    if not np.any(mask):
        return f_nominal   # safety fallback

    # Find peak bin within the search window
    local_spec          = np.where(mask, spec, 0.0)
    peak_idx            = int(np.argmax(local_spec))

    # Parabolic interpolation for sub-bin precision
    if 0 < peak_idx < len(spec) - 1:
        alpha    = spec[peak_idx - 1]
        beta     = spec[peak_idx]
        gamma    = spec[peak_idx + 1]
        denom    = alpha - 2.0 * beta + gamma
        if denom != 0.0:
            correction = 0.5 * (alpha - gamma) / denom
        else:
            correction = 0.0
        bin_width    = freqs[1] - freqs[0]
        actual_freq  = freqs[peak_idx] + correction * bin_width
    else:
        actual_freq  = freqs[peak_idx]

    # Final sanity check — result must stay inside the search window
    if abs(actual_freq - f_nominal) > search_range:
        return f_nominal

    return float(actual_freq)


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAND MEASUREMENT  — AM-IM at carrier ± f_low
#  Matches REW d2L / d2H values directly.
#  Uses FFT peak with ±3 bin search + parabolic interpolation.
# ═════════════════════════════════════════════════════════════════════════════

def measure_sidebands(audio: np.ndarray, fs: float,
                      f_low: float, f_high: float) -> dict:
    """
    Measure first-order AM sidebands at f_high ± f_low using FFT peak.

    Returns dict with:
      usb_pct   : upper sideband amplitude as % of carrier  (REW d2H equivalent)
      lsb_pct   : lower sideband amplitude as % of carrier  (REW d2L equivalent)
      carrier_db: carrier level (dBFS)
      usb_db    : upper sideband (dBFS)
      lsb_db    : lower sideband (dBFS)
      imd_pct   : Bauer IM% = avg(usb_pct, lsb_pct)
    """
    N        = len(audio)
    win      = np.hanning(N)
    win_corr = 2.0 / np.mean(win)
    spec     = np.abs(np.fft.rfft(audio * win)) * win_corr / N
    freqs    = np.fft.rfftfreq(N, 1.0 / fs)
    bin_hz   = freqs[1]

    def fft_peak(f_target):
        idx_nom = int(round(f_target / bin_hz))
        lo      = max(1, idx_nom - 3)
        hi      = min(len(spec) - 2, idx_nom + 3)
        idx     = lo + int(np.argmax(spec[lo:hi+1]))
        a, b, g = spec[idx-1], spec[idx], spec[idx+1]
        denom   = a - 2*b + g
        corr    = 0.5*(a-g)/denom if denom != 0 else 0.0
        return max(float(b + 0.5*corr*(g-a)), 0.0)

    car_v = fft_peak(f_high)
    usb_v = fft_peak(f_high + f_low)
    lsb_v = fft_peak(f_high - f_low)

    def _db(v): return float(20*np.log10(max(v, 1e-12)))

    if car_v < 1e-9:
        return dict(usb_pct=0.0, lsb_pct=0.0, imd_pct=0.0,
                    carrier_db=-120.0, usb_db=-120.0, lsb_db=-120.0)

    usb_pct = 100.0 * usb_v / car_v
    lsb_pct = 100.0 * lsb_v / car_v
    imd_pct = (usb_pct + lsb_pct) / 2.0   # Bauer definition

    return dict(
        usb_pct    = usb_pct,
        lsb_pct    = lsb_pct,
        imd_pct    = imd_pct,
        carrier_db = _db(car_v),
        usb_db     = _db(usb_v),
        lsb_db     = _db(lsb_v),
    )




def analyse(audio: np.ndarray, fs: float,
            channel: str   = 'L',
            f_low: float   = F_MOD,
            f_high: float  = F_CARRIER,
            R: float       = GROOVE_RADIUS,
            V_low: float   = PEAK_VEL,
            theta_r: float = RECORDING_ANGLE,
            swap_channels: bool = False,
            modulation: str = MODULATION,
            verbose: bool  = True) -> dict:
    """Run the full patent signal chain on one channel of audio."""

    # ── Auto-detect actual tone frequencies ──────────────────────────────
    # Turntable speed error or non-standard cutting frequencies can shift
    # the actual tones away from their nominal values.  Even a 1 Hz offset
    # on f_low causes a continuous phase rotation in the chopper reference,
    # making φ (and therefore HTA/VTA) unreliable.
    # We detect both tones from the spectrum and use the actual frequencies
    # everywhere downstream — filters, chopper reference, FM discriminator.
    #
    # Pre-filter before detection to isolate each tone cleanly:
    #   f_low  : bandpass ±15 % around nominal f_low
    #   f_high : highpass above f_low×3 to remove the modulating tone
    _audio_low  = apply(_sos_bp(fs, f_low,  f_low  * 0.30, order=4), audio)
    _audio_high = apply(_sos_hp(fs, f_low * 3.0,            order=4), audio)

    f_low_actual  = detect_actual_frequency(_audio_low,  fs, f_low,
                                             search_range=f_low  * 0.05)
    f_high_actual = detect_actual_frequency(_audio_high, fs, f_high,
                                             search_range=f_high * 0.05)

    if verbose:
        offset_low  = f_low_actual  - f_low
        offset_high = f_high_actual - f_high
        flag_low    = "  ← speed error!" if abs(offset_low)  > 1.0 else ""
        flag_high   = "  ← speed error!" if abs(offset_high) > 1.0 else ""
        print(f"  [Freq] f_low  nominal={f_low:.1f} Hz  "
              f"detected={f_low_actual:.2f} Hz  "
              f"({offset_low:+.2f} Hz){flag_low}")
        print(f"  [Freq] f_high nominal={f_high:.1f} Hz  "
              f"detected={f_high_actual:.2f} Hz  "
              f"({offset_high:+.2f} Hz){flag_high}")

    # Use detected frequencies for all downstream processing
    f_low  = f_low_actual
    f_high = f_high_actual

    # ── Input level check + RIAA phase warning ───────────────────────────
    # RIAA amplitude effect: shifts f_low/f_high ratio by ~+11 to +15 dB.
    # More critically, RIAA introduces a PHASE SHIFT of ~50-60° at f_low.
    # This corrupts the chopper reference (Path B) relative to the FM
    # deviation signal (Path A), rotating phi and potentially flipping the
    # sign of HTA/VTA on one channel — verified experimentally:
    #   With RIAA:    L=-2.8°, R=+3.4°  (opposite signs — WRONG)
    #   Without RIAA: L=+2.7°, R=+3.4°  (same sign    — CORRECT)
    # The FM deviation amplitude is RIAA-immune, but the phase is not.
    # Always record with FLAT/BYPASS preamp for this measurement.

    BW_LOW  = f_low  * 0.30
    BW_HIGH = f_high * 0.30

    e_low  = float(np.std(apply(_sos_bp(fs, f_low,  BW_LOW,  order=4), audio)))
    e_high = float(np.std(apply(_sos_bp(fs, f_high, BW_HIGH, order=4), audio)))

    _riaa_ratio_db = None

    if e_high > 0 and e_low > 0:
        ratio_db       = 20.0 * np.log10(e_low / e_high)
        _riaa_ratio_db = ratio_db
        if ratio_db > 20.0:
            warnings.warn(
                f"\n"
                f"⚠  POSSIBLE RIAA EQUALIZATION DETECTED\n"
                f"   {f_low:.0f}Hz / {f_high:.0f}Hz amplitude ratio = {ratio_db:.1f} dB\n"
                f"\n"
                f"   RIAA introduces a ~50-60° phase shift at {f_low:.0f}Hz.\n"
                f"   This corrupts the chopper reference phase (Path B) and\n"
                f"   can flip the HTA/VTA sign on one channel, giving wrong results:\n"
                f"\n"
                f"     With RIAA:    L and R may show OPPOSITE signs  (WRONG)\n"
                f"     Without RIAA: L and R show the SAME sign       (CORRECT)\n"
                f"\n"
                f"   The FM deviation amplitude is RIAA-immune, but phase is not.\n"
                f"   ACTION: Re-record using your preamp FLAT / BYPASS output.",
                UserWarning, stacklevel=2)
        elif e_low < 1e-6:
            if verbose:
                print(f"  [!] Level check: {f_low:.0f}Hz tone not detected — "
                      f"wrong track or input disconnected?")
        elif e_high < 1e-6:
            if verbose:
                print(f"  [!] Level check: {f_high:.0f}Hz carrier not detected — "
                      f"wrong track or input disconnected?")
        else:
            if verbose:
                status = "OK" if -10 <= ratio_db <= 20 else "CHECK"
                print(f"  [{status}] Input level check: "
                      f"{f_low:.0f}Hz/{f_high:.0f}Hz ratio = {ratio_db:.1f} dB")
    elif verbose:
        print(f"  [!] Level check: could not detect tones — "
              f"check input connections and recording level")

    # ── Path A: FM discriminator ──────────────────────────────────────────
    dev_signal  = path_a(audio, fs, f_low, f_high)

    # ── Chopper chain — TRACKING mode (0°) ───────────────────────────────
    sq_track    = path_b(audio, fs, f_low, f_high, mode='tracking')
    dc_track, chopped_track, dc_track_sig = chopper_lpf(dev_signal, sq_track, fs, f_low)

    # ── Chopper chain — TRACING mode (90°) ───────────────────────────────
    sq_trace    = path_b(audio, fs, f_low, f_high, mode='tracing')
    dc_trace, chopped_trace, dc_trace_sig = chopper_lpf(dev_signal, sq_trace, fs, f_low)

    # ── Extract F (peak deviation, unsigned) ─────────────────────────────
    F_peak, phi_deg = extract_F_and_phi(dev_signal, audio, fs, f_low)

    # ── Low-F sanity check — catches wrong TONE_PAIR setting ─────────────
    # If F is near zero despite tones being present, the most likely cause
    # is that TONE_PAIR does not match the actual record track.
    # Example: TONE_PAIR='200+4000' set while playing a 400+4000 Hz band —
    # the 200Hz bandpass finds nothing, the discriminator sees only noise,
    # and F collapses to < 1 Hz.  The tracing distortion harmonics then
    # show nonsensical values (including positive dB re fundamental) because
    # they are computing ratios of noise to noise.
    # Threshold: 5 Hz is well above the noise floor (~0.5 Hz) but well
    # below any real tracking/tracing signal (typically 20-200 Hz).
    _F_MIN_VALID = 5.0   # Hz
    if F_peak < _F_MIN_VALID and (e_low > 1e-6 and e_high > 1e-6):
        warnings.warn(
            f"\n"
            f"⚠  FM DEVIATION TOO LOW  (channel {channel}: F = {F_peak:.2f} Hz)\n"
            f"\n"
            f"   F < {_F_MIN_VALID:.0f} Hz despite both tones being present in the recording.\n"
            f"   This almost always means TONE_PAIR does not match the actual\n"
            f"   track on the record.\n"
            f"\n"
            f"   Detected tones:  f_low={f_low:.0f} Hz,  f_high={f_high:.0f} Hz\n"
            f"   Current setting: TONE_PAIR='{TONE_PAIR}' → expects f_low≈{F_MOD:.0f} Hz\n"
            f"\n"
            f"   Likely fixes:\n"
            f"   • Playing a 400+4000 Hz band?  Set TONE_PAIR = '400+4000'\n"
            f"   • Playing a 200+4000 Hz band?  Set TONE_PAIR = '200+4000'\n"
            f"   • Custom record?               Set TONE_PAIR = 'custom' and\n"
            f"                                  CUSTOM_F_MOD to the actual f_low\n"
            f"\n"
            f"   VTA/HTA result and tracing distortion plot will be meaningless.",
            UserWarning, stacklevel=2)

    # ── F_signed: sign from most stable segments ──────────────────────────
    # Split into 2-second segments, rank by |DC| magnitude, take mean of
    # top 50% — most settled/signal-rich segments determine the sign.
    # Falls back to overall dc_track if signal is too short for segmenting.
    seg_len_sign = int(fs * 2.0)
    dc_segs      = []
    dc_top_mean  = 0.0   # initialise here so verbose print never crashes

    # Only segment if we have at least 2 full segments
    if len(audio) >= seg_len_sign * 2:
        n_segs_sign = len(audio) // seg_len_sign
        for i in range(n_segs_sign):
            sl  = slice(i * seg_len_sign, (i + 1) * seg_len_sign)
            seg = audio[sl]
            if len(seg) < seg_len_sign:
                continue
            dev_s = path_a(seg, fs, f_low, f_high)
            sq_s  = path_b(seg, fs, f_low, f_high, mode='tracking')
            dc_s, _, _ = chopper_lpf(dev_s, sq_s, fs, f_low)
            if abs(dc_s) > 0.5:
                dc_segs.append(dc_s)

    if dc_segs:
        dc_sorted   = sorted(dc_segs, key=abs, reverse=True)
        top_half    = dc_sorted[:max(1, len(dc_sorted) // 2)]
        dc_top_mean = float(np.mean(top_half))
        dc_sign_robust = np.sign(dc_top_mean)
    else:
        # Short signal or all segments near zero — use overall dc_track directly
        dc_top_mean    = dc_track
        dc_sign_robust = np.sign(dc_track)

    # Channel polarity: left channel inverts vertical modulation in groove
    # SWAP_CHANNELS flips this if the recording has L/R wired in reverse
    # Channel polarity — White & Gust (1979) Table 1:
    #
    # VERTICAL modulation (Group 2B, theta_r ≈ 16°):
    #   L and R groove walls move OPPOSITE directions for vertical modulation.
    #   Chopper DC has opposite sign on L vs R → ch_sign=-1 for left.
    #   Left:  θPV = arctan(tan(θR) - C·R·F·cos(φ)/V_sv)  ← minus
    #   Right: θPV = arctan(tan(θR) + C·R·F·cos(φ)/V_sv)  ← plus
    #
    # LATERAL modulation (Group 1B, theta_r = 0°):
    #   L and R groove walls move the SAME direction for lateral modulation.
    #   Chopper DC has the SAME sign on both channels → ch_sign=+1 for both.
    #   Both: θPH = 8.72e-4·R·F·cos(φ)/V_SH  (identical formula L and R)
    #   This gives same sign and similar magnitude on L and R, as shown in
    #   White & Gust Fig. 3 where both channels read ~+2°.
    #
    # SWAP_CHANNELS flips ch_is_left if L/R wiring is physically reversed.
    ch_is_left  = (channel.upper() == 'L')
    if swap_channels:
        ch_is_left = not ch_is_left
    _is_lateral = (modulation == "lateral")
    ch_sign     = +1.0 if _is_lateral else (-1.0 if ch_is_left else +1.0)
    F_signed    = ch_sign * dc_sign_robust * F_peak

    if verbose:
        n_total   = len(dc_segs)
        n_correct = sum(1 for d in dc_segs if np.sign(d) == dc_sign_robust)
        top_n     = max(1, n_total // 2)
        if n_total == 0:
            print(f"  [Sign] using overall DC={dc_top_mean:+.1f}  "
                  f"(signal too short for segment analysis)")
        else:
            stability = 'stable' if n_total == 0 or n_correct/n_total > 0.6 \
                        else 'check recording'
            print(f"  [Sign] top-{top_n} segments mean DC={dc_top_mean:+.1f}  "
                  f"{n_correct}/{n_total} segments agree  ({stability})")

    # ── RIAA phase check ─────────────────────────────────────────────────
    # RIAA phase shift at f_low rotates phi by ~50-60°.  If phi is large
    # (|phi| > 70°) the real component F·cos(phi) is substantially reduced
    # and the sign may be flipped relative to a flat recording.
    # This check fires even when the amplitude ratio didn't trigger the
    # earlier warning — e.g. RIAA records played through RIAA preamp have
    # a normal amplitude ratio but still suffer from phase corruption.
    _phi_riaa_threshold = 70.0
    if abs(phi_deg) > _phi_riaa_threshold:
        warnings.warn(
            f"\n"
            f"⚠  LARGE PHASE ANGLE DETECTED  (channel {channel}: φ = {phi_deg:+.1f}°)\n"
            f"\n"
            f"   |φ| > {_phi_riaa_threshold:.0f}° means the FM deviation and the {f_low:.0f}Hz reference\n"
            f"   are substantially out of phase.  Most likely causes:\n"
            f"\n"
            f"   1. RIAA equalization applied during recording (most common).\n"
            f"      RIAA introduces ~50-60° phase shift at {f_low:.0f}Hz, rotating\n"
            f"      φ away from 0° and reducing or inverting the real component.\n"
            f"      FIX: Re-record with preamp set to FLAT / BYPASS.\n"
            f"\n"
            f"   2. High tracing distortion (large imaginary FM component).\n"
            f"      This is a real physical effect and does not indicate an\n"
            f"      error — but the HTA/VTA result may still be valid if both\n"
            f"      channels give the same sign.\n"
            f"\n"
            f"   If L and R channels show OPPOSITE signs, cause (1) is likely.\n"
            f"   If L and R channels show the SAME sign, cause (2) is likely.",
            UserWarning, stacklevel=2)

    # ── VTA computation (patent formula) ─────────────────────────────────
    # The chopper [58] implicitly computes F·cos(φ) by multiplying the
    # deviation signal by the reference square wave — this extracts only
    # the real (tracking) component of FM distortion. F_signed already
    # encodes this; no separate cos(φ) correction is needed here.
    vta     = compute_vta(F_signed, R, V_low, theta_r, f_low, f_high, modulation=modulation)
    vta_err = vta - theta_r

    # ── Tracing distortion harmonics ─────────────────────────────────────
    harmonics = tracing_distortion(dev_signal, fs, f_low)

    # ── AM sideband measurement (Bauer AM-IM, REW d2L/d2H equivalent) ────
    sidebands = measure_sidebands(audio, fs, f_low, f_high)

    # ── DC tracking signal (offset-corrected, as meter [62]) ─────────────
    #   offset sets zero-deviation point at θR (16.0° for CBS STR-112)
    meter_reading = vta   # in degrees, directly

    results = dict(
        channel          = channel,
        modulation       = modulation,
        f_low            = f_low,
        f_high           = f_high,
        f_low_nominal    = F_MOD,
        f_high_nominal   = F_CARRIER,
        fs               = fs,
        F_peak_hz     = F_peak,
        F_signed_hz   = F_signed,
        phi_deg       = phi_deg,
        vta_deg       = vta,
        vta_error_deg = vta_err,
        theta_r_deg   = theta_r,
        dc_tracking   = dc_track,
        dc_tracing    = dc_trace,
        harmonics_db  = harmonics,
        dev_signal    = dev_signal,
        sq_track      = sq_track,
        chopped_track = chopped_track,
        dc_track_sig  = dc_track_sig,
        chopped_trace = chopped_trace,
        dc_trace_sig  = dc_trace_sig,
        audio         = audio,
        sidebands     = sidebands,    # AM-IM: usb_pct, lsb_pct, imd_pct
    )

    if verbose:
        bar_w = 30
        print(f"\n╔══════════════════════════════════════════════════╗")
        _anlbl = "HTA ANALYSIS" if modulation == "lateral" else "VTA ANALYSIS"
        print(f"║  {_anlbl} — Channel {channel}  ({f_low:.0f}/{f_high:.0f} Hz)         ║")
        print(f"╠══════════════════════════════════════════════════╣")
        print(f"║  Peak FM deviation (F)    : {F_peak:8.2f} Hz             ║")
        print(f"║  Signed FM deviation      : {F_signed:+8.2f} Hz             ║")
        print(f"║  Phase lead E2/E1 (φ)     : {phi_deg:+8.1f}°             ║")
        _anglbl = "Horiz Tracking Angle " if modulation == "lateral" else "Vertical Tracking Angle"
        print(f"║  {_anglbl}  : {vta:8.2f}°             ║")
        _errlbl = "HTA" if modulation == "lateral" else "VTA"
        print(f"║  {_errlbl} Error (vs θR={theta_r:.1f}°) : {vta_err:+8.2f}°             ║")
        print(f"║  DC tracking component    : {dc_track:+8.4f}               ║")
        print(f"║  DC tracing  component    : {dc_trace:+8.4f}               ║")
        print(f"╠══════════════════════════════════════════════════╣")
        print(f"║  Wow/flutter tolerance:                          ║")
        print(f"║  Wow (<3Hz) and flutter (>10Hz offset) are fully ║")
        print(f"║  rejected by the 400Hz BPF and chopper averaging.║")
        print(f"║  Flutter within ~10Hz of {f_low:.0f}Hz adds <0.5deg error.  ║")
        print(f"╠══════════════════════════════════════════════════╣")
        print(f"║  Tracing distortion harmonics (dB re fundamental)║")
        for n, db in harmonics.items():
            filled = int(max(0, (db + 60)/2))
            bar    = '█'*filled + '░'*(bar_w-filled)
            print(f"║  {n:2d}×{f_low:.0f}Hz : {db:+6.1f} dB  {bar[:20]}  ║")
        print(f"╚══════════════════════════════════════════════════╝")

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC TEST SIGNAL GENERATOR  (calibration / demo)
# ═════════════════════════════════════════════════════════════════════════════

def generate_test_signal(fs: float = 44100.0,
                         duration: float = 10.0,
                         vta_error_deg: float = 6.0,
                         channel: str = 'L',
                         f_low: float  = F_MOD,
                         f_high: float = F_CARRIER,
                         tracing_frac: float = 0.02,
                         imd_factor: float = 3.0) -> np.ndarray:
    """
    Synthesize a CBS STR-112-style test signal with a known VTA error injected.

    The signal contains two physically correct components:

    1. FM deviation on the 4kHz carrier — encodes VTA error for the
       White & Gust FM discriminator method (patent US 4,359,768).
       This is what compute_vta() reads.

    2. AM sidebands on the 4kHz carrier — encodes Bauer AM-IM distortion.
       Real groove physics produces both FM and AM simultaneously.
       The AM level is set using the Bauer exact formula (Eq.24) multiplied
       by imd_factor to match empirical data (real records show ~3× higher
       IM% than the pure tracking formula predicts, due to tracing distortion
       and other groove-stylus interaction effects).

    Parameters
    ----------
    vta_error_deg : float  — VTA error above recording angle (degrees)
    imd_factor    : float  — multiplier on Bauer IM% to match real data
                             (default 3.0 — calibrated against real STR-111
                              measurements where 6° error → ~6% IM%)
    """
    t = np.arange(int(fs * duration)) / fs

    # ── Channel polarity ─────────────────────────────────────────────────
    groove_sign = -1.0 if channel.upper() == 'L' else +1.0

    # ── FM deviation — encodes VTA error for discriminator method ─────────
    # Use a small representative VTA error for FM (the discriminator reads this).
    # We keep FM small so its sidebands (J1(β) ≈ β/2) don't swamp the AM
    # sideband measurement. On real records FM and AM coexist at the same
    # frequencies and cannot be separated by FFT.
    fm_error_deg = min(vta_error_deg, 1.5)   # cap FM contribution
    theta_P_fm  = np.radians(RECORDING_ANGLE + fm_error_deg)
    theta_R     = np.radians(RECORDING_ANGLE)
    C_gen       = 2.0*np.pi*TURNTABLE_RPM / (60.0*f_high*np.cos(np.radians(RECORDING_ANGLE)))
    F_inj       = (PEAK_VEL * (np.tan(theta_P_fm) - np.tan(theta_R))) / (C_gen * GROOVE_RADIUS)

    mod_idx_track = groove_sign * F_inj / f_low
    mod_idx_trace = tracing_frac * abs(mod_idx_track)

    carrier_phase = (2*np.pi*f_high*t
                     + mod_idx_track * np.sin(2*np.pi*f_low*t)      # FM tracking
                     + mod_idx_trace * (-np.cos(2*np.pi*f_low*t)))  # FM tracing

    # ── AM sidebands — encodes Bauer IM% ─────────────────────────────────
    # Bauer exact Eq.24: IM = (v/V)*cos(C)*(tan(A)-tan(C))
    # Uses full vta_error_deg so the AM level reflects the intended error.
    # imd_factor accounts for real-world tracing + groove effects (~3×).
    theta_P_am = np.radians(RECORDING_ANGLE + vta_error_deg)
    C_rec      = np.radians(RECORDING_ANGLE)
    vV         = PEAK_VEL / (2*np.pi*GROOVE_RADIUS*(TURNTABLE_RPM/60.0))
    im_bauer   = vV * np.cos(C_rec) * (np.tan(theta_P_am) - np.tan(theta_R))
    m_am       = 2.0 * im_bauer * imd_factor   # AM modulation index

    carrier_amp = 0.15
    carrier     = carrier_amp * (1.0 + m_am * np.cos(2*np.pi*f_low*t)) \
                  * np.cos(carrier_phase)

    # ── 400Hz modulating tone (E1 reference) ─────────────────────────────
    mod_tone = 0.7 * np.sin(2*np.pi*f_low*t)

    combined = mod_tone + carrier
    pk = np.max(np.abs(combined))
    if pk > 0:
        combined /= pk * 1.05

    return combined.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
#  PLOTTING  — replicates meter face + signal chain visualisation
# ═════════════════════════════════════════════════════════════════════════════

DARK   = '#0e1117'
MID    = '#1c2230'
ACCENT = '#00d4ff'
GREEN  = '#39ff14'
AMBER  = '#ffb300'
RED    = '#ff3d3d'
WHITE  = '#e8eaf0'
GREY   = '#5a6070'

def _style_ax(ax, title=''):
    ax.set_facecolor(MID)
    for sp in ax.spines.values():
        sp.set_color(WHITE)
    ax.tick_params(colors=WHITE, labelsize=10)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    if title:
        ax.set_title(title, color=WHITE, fontsize=11, pad=6)


def plot_signal_chain(res: dict):
    """Show each stage of the patent signal chain for one channel."""
    fs   = res['fs']
    show = min(int(fs * 0.025), len(res['audio']))  # 25 ms
    t    = np.arange(show) / fs * 1000              # ms

    fig  = plt.figure(figsize=(14, 9), facecolor=DARK)
    fig.suptitle(f"Patent Signal Chain — Channel {res['channel']}  "
                 f"({res['f_low']:.0f}/{res['f_high']:.0f} Hz)",
                 color=WHITE, fontsize=15, fontweight='bold')

    stages = [
        ('Input (audio)',        res['audio'][:show],         ACCENT),
        ('Path A: FM deviation', res['dev_signal'][:show],    AMBER),
        ('Path B: square wave',  res['sq_track'][:show],      GREEN),
        ('Chopper out (track)',   res['chopped_track'][:show], '#ff9800'),
        ('LPF → DC (tracking)',  res['dc_track_sig'][:show],  GREEN),
        ('Chopper out (trace)',   res['chopped_trace'][:show], '#e040fb'),
        ('LPF → DC (tracing)',   res['dc_trace_sig'][:show],  '#e040fb'),
    ]

    for i, (label, data, col) in enumerate(stages):
        ax = fig.add_subplot(len(stages), 1, i+1)
        ax.plot(t, data, color=col, linewidth=0.9, alpha=0.92)
        ax.axhline(0, color=WHITE, linewidth=0.4, linestyle='--', alpha=0.3)
        _style_ax(ax, label)
        ax.set_xlim(0, t[-1])
        if i == len(stages)-1:
            ax.set_xlabel('Time (ms)', color=WHITE, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_meter_dashboard(results_list: list,
                         script_name: str = '',
                         wav_name: str = '',
                         description: str = ''):
    """Meter face (Fig. 5) + AM sidebands + harmonics for all analysed results."""
    n    = len(results_list)
    fig  = plt.figure(figsize=(5*n, 14), facecolor=DARK)

    # ── Main title ────────────────────────────────────────────────────────
    fig.suptitle('Vertical Tracking Angle Meter  —  US Patent 4,359,768',
                 color=WHITE, fontsize=16, fontweight='bold', y=0.98)

    # ── Info banner: script / WAV / description ───────────────────────────
    import os as _os
    script_short = _os.path.basename(script_name) if script_name else '(unknown)'
    wav_short    = _os.path.basename(wav_name)    if wav_name    else '(demo/synthetic)'
    info_lines   = (f"Script: {script_short}     WAV: {wav_short}")
    if description:
        info_lines += f"\n{description}"

    fig.text(0.5, 0.955, info_lines,
             ha='center', va='top', color=ACCENT,
             fontsize=14, fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=MID,
                       edgecolor=GREY, linewidth=0.8))

    # ── Average VTA label (midway between meter circles) ─────────────────
    if len(results_list) >= 2:
        vta_vals  = [r['vta_deg'] for r in results_list]
        vta_avg   = float(np.mean(vta_vals))
        theta_r_0 = results_list[0]['theta_r_deg']
        err_avg   = vta_avg - theta_r_0
        avg_col   = GREEN if abs(err_avg) < 2 else AMBER if abs(err_avg) < 5 else RED
        _mode_lbl = "HTA" if any(r.get('modulation','vertical')=='lateral'
                                  for r in results_list) else "VTA"
        avg_label = f"AVG {_mode_lbl}\n{vta_avg:.2f}°\nDev {err_avg:+.2f}°"
        fig.text(0.5, 0.575, avg_label,
                 ha='center', va='center',
                 color='#111111', fontsize=16, fontweight='bold',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='#FFD700',
                           edgecolor=avg_col, linewidth=3.0),
                 zorder=20)

    # ── Average IM% label (below AVG VTA, above sideband bars) ───────────
    if len(results_list) >= 1:
        all_sb = [r.get('sidebands', {}) for r in results_list]
        pcts   = ([sb.get('usb_pct', 0) for sb in all_sb] +
                  [sb.get('lsb_pct', 0) for sb in all_sb])
        pcts   = [p for p in pcts if p > 0]
        if pcts:
            avg_imd = float(np.mean(pcts))
            imd_col = GREEN if avg_imd < 3 else AMBER if avg_imd < 7 else RED
            fig.text(0.5, 0.390,
                     f"AVG IM%\n(4 sidebands)\n{avg_imd:.2f}%",
                     ha='center', va='center',
                     color='#111111', fontsize=12, fontweight='bold',
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.4',
                               facecolor=imd_col,
                               edgecolor='#333333', linewidth=2.0),
                     zorder=20)

    # ── GridSpec: 3 rows — meter, sideband bars, harmonic bars ───────────
    for col_idx, res in enumerate(results_list):
        gs = gridspec.GridSpec(3, n, figure=fig,
                               left=0.06, right=0.97,
                               top=0.84, bottom=0.04,
                               hspace=0.55, wspace=0.4,
                               height_ratios=[1.8, 0.9, 0.9])

        # ── Meter face (top row) ──────────────────────────────────────────
        ax_m = fig.add_subplot(gs[0, col_idx], projection='polar')
        ax_m.set_facecolor(MID)
        ax_m.set_theta_direction(-1)
        ax_m.set_theta_offset(np.pi)
        ax_m.set_ylim(0, 1.15)
        ax_m.set_yticks([])

        vta_min, vta_max = 5.0, 40.0
        vta_range = vta_max - vta_min

        def vta_to_rad(v):
            return np.pi * (v - vta_min) / vta_range

        zones = [(5, 13, RED), (13, 17, AMBER), (17, 25, GREEN),
                 (25, 33, AMBER), (33, 40, RED)]
        for z0, z1, zcol in zones:
            th = np.linspace(vta_to_rad(z0), vta_to_rad(z1), 60)
            ax_m.plot(th, np.ones(60)*1.0, color=zcol, linewidth=8,
                      solid_capstyle='butt', alpha=0.7)

        for deg in range(5, 41, 5):
            th = vta_to_rad(deg)
            ax_m.plot([th, th], [0.85, 1.0], color=WHITE, linewidth=0.8)
            ax_m.text(th, 1.12, f'{deg}°', ha='center', va='center',
                      color=WHITE, fontsize=12,
                      rotation=np.degrees(th) - 90)

        th_r = vta_to_rad(res['theta_r_deg'])
        ax_m.plot([th_r, th_r], [0.75, 1.03], color=ACCENT,
                  linewidth=1.2, linestyle='--', alpha=0.7)
        ax_m.text(th_r, 0.65, f"θR={res['theta_r_deg']}°",
                  ha='center', color=ACCENT, fontsize=12)

        vta_clamped = np.clip(res['vta_deg'], vta_min, vta_max)
        th_n = vta_to_rad(vta_clamped)
        needle_col = GREEN if abs(res['vta_error_deg']) < 2 else \
                     AMBER  if abs(res['vta_error_deg']) < 5 else RED
        ax_m.annotate('', xy=(th_n, 0.92), xytext=(th_n, 0.0),
                      arrowprops=dict(arrowstyle='->', color=needle_col,
                                      lw=2.5, mutation_scale=12))
        ax_m.plot(0, 0, 'o', color=WHITE, markersize=5, zorder=10)

        _m_lbl = "HTA" if res.get('modulation','vertical') == 'lateral' else "VTA"
        ax_m.set_title(
            f"Ch {res['channel']}  {res['f_low']:.0f}/{res['f_high']:.0f} Hz\n"
            f"{_m_lbl} = {res['vta_deg']:.1f}°  (err {res['vta_error_deg']:+.1f}°)\n"
            f"F={res['F_peak_hz']:.1f} Hz  φ={res['phi_deg']:+.0f}°",
            color=WHITE, fontsize=12, pad=14)

        ax_m.set_rticks([])
        ax_m.set_thetagrids([])
        ax_m.spines['polar'].set_visible(False)

        # ── AM sideband bars (middle row) ─────────────────────────────────
        ax_s = fig.add_subplot(gs[1, col_idx])
        ax_s.set_facecolor(MID)

        sb = res.get('sidebands', {})
        usb_pct = sb.get('usb_pct', 0.0)
        lsb_pct = sb.get('lsb_pct', 0.0)
        imd_pct = sb.get('imd_pct', 0.0)
        f_low_a = res['f_low']
        f_high_a = res['f_high']

        bar_labels = [f'LSB\n{f_high_a:.0f}−{f_low_a:.0f}',
                      f'USB\n{f_high_a:.0f}+{f_low_a:.0f}']
        bar_vals   = [lsb_pct, usb_pct]
        bar_cols   = ['#00bcd4', '#ff6ec7']   # cyan LSB, pink USB

        bars = ax_s.bar([0, 1], bar_vals, color=bar_cols,
                        edgecolor=DARK, width=0.5)
        ax_s.set_xticks([0, 1])
        ax_s.set_xticklabels(bar_labels, color=WHITE, fontsize=9)
        ax_s.set_ylabel('% of carrier', color=WHITE, fontsize=9)
        ax_s.axhline(0, color=WHITE, linewidth=0.4, alpha=0.3)

        # Value labels — inside each bar near the top to avoid overlap
        for bar, val in zip(bars, bar_vals):
            ax_s.text(bar.get_x() + bar.get_width()/2,
                      val * 0.6,
                      f'{val:.2f}%', ha='center', va='center',
                      color='#111111', fontsize=9, fontweight='bold')

        # IM% label — below the bars as a subtitle, not overlapping
        imd_col_s = GREEN if imd_pct < 3 else AMBER if imd_pct < 7 else RED
        ax_s.text(0.5, -0.22,
                  f'IM% = {imd_pct:.2f}%  (avg LSB+USB)',
                  transform=ax_s.transAxes, ha='center', va='top',
                  color=imd_col_s, fontsize=8, fontweight='bold',
                  fontfamily='monospace')

        _style_ax(ax_s, f'AM Sidebands — Ch {res["channel"]}  '
                        f'(d2L={lsb_pct:.2f}%  d2H={usb_pct:.2f}%)')

        # ── Tracing distortion harmonics (bottom row) ─────────────────────
        ax_h = fig.add_subplot(gs[2, col_idx])
        ax_h.set_facecolor(MID)
        harms  = list(res['harmonics_db'].keys())
        dbvals = [res['harmonics_db'][h] for h in harms]
        cmap   = plt.cm.plasma
        colors = [cmap(h/max(harms)) for h in harms]
        ax_h.bar(harms, dbvals, color=colors, edgecolor=DARK, width=0.6)
        ax_h.axhline(-20, color=AMBER, linewidth=0.8, linestyle='--',
                     alpha=0.7, label='−20 dB')
        ax_h.axhline(-40, color=GREEN, linewidth=0.8, linestyle='--',
                     alpha=0.7, label='−40 dB')
        ax_h.set_xlabel('Harmonic', color=WHITE, fontsize=9)
        ax_h.set_ylabel('dB re fundamental', color=WHITE, fontsize=9)
        ax_h.set_xticks(harms)
        ax_h.set_xticklabels([f'{n}×' for n in harms])
        ax_h.legend(fontsize=9, facecolor=MID, edgecolor=MID,
                    labelcolor=WHITE, loc='lower right')
        _style_ax(ax_h, f'Tracing Distortion — {res["f_low"]:.0f}Hz harmonics')

    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  SPECTRUM PLOT  — verify tone detection and signal quality
# ═════════════════════════════════════════════════════════════════════════════

def plot_spectrum(audio_map: dict, fs: float,
                 f_low: float = F_MOD,
                 f_high: float = F_CARRIER) -> plt.Figure:
    """
    FFT spectrum for all channels (L and/or R) shown side by side.
    Top row    : full spectrum 0 – f_high*1.5, tones annotated.
    Bottom row : zoom around the carrier showing FM sidebands.
    audio_map  : dict  {'L': array, 'R': array}  — one or both channels.
    """
    channels  = list(audio_map.keys())
    n_ch      = len(channels)
    ch_colors = {'L': ACCENT, 'R': '#ff6ec7'}   # cyan for L, pink for R

    fig, axes = plt.subplots(2, n_ch,
                             figsize=(7 * n_ch, 9),
                             facecolor=DARK,
                             squeeze=False)
    fig.suptitle(f'Input Spectrum  —  {f_low:.0f} Hz / {f_high:.0f} Hz',
                 color=WHITE, fontsize=15, fontweight='bold')

    zoom_bw  = max(f_low * 4, 600)   # ±zoom_bw Hz around carrier
    f_max    = min(f_high * 1.6, fs / 2 - 1)

    for col, ch in enumerate(channels):
        audio = audio_map[ch]
        col_color = ch_colors.get(ch, ACCENT)

        N     = len(audio)
        freqs = np.fft.rfftfreq(N, 1 / fs)
        spec  = 20 * np.log10(
                    np.abs(np.fft.rfft(audio * np.hanning(N))) * 2 / N + 1e-12)

        def find_peak_db(fc, bw=None):
            # Default window: ±15% of centre frequency
            # Robust to turntable speed errors and works for all tone pairs
            if bw is None:
                bw = fc * 0.15
            m = (freqs >= fc - bw) & (freqs <= fc + bw)
            return float(np.max(spec[m])) if np.any(m) else -120.0

        # ── Top: full spectrum ────────────────────────────────────────────
        ax0  = axes[0, col]
        ax0.set_facecolor(MID)
        mask = freqs <= f_max
        ax0.plot(freqs[mask], spec[mask],
                 color=col_color, linewidth=0.8, alpha=0.92)
        ax0.axvline(f_low,  color=GREEN, linewidth=1.2, linestyle='--',
                    label=f'{f_low:.0f} Hz (mod tone)')
        ax0.axvline(f_high, color=AMBER, linewidth=1.2, linestyle='--',
                    label=f'{f_high:.0f} Hz (carrier)')
        ax0.set_xlabel('Frequency (Hz)', color=WHITE, fontsize=11)
        ax0.set_ylabel('Amplitude (dBFS)', color=WHITE, fontsize=11)
        ax0.legend(fontsize=10, facecolor=MID, edgecolor=MID, labelcolor=WHITE)
        _style_ax(ax0, f'Channel {ch}  —  Full Spectrum')

        # Annotate tone peak levels and ratio
        db_low  = find_peak_db(f_low)
        db_high = find_peak_db(f_high)
        ratio   = db_low - db_high
        # Place annotation to the right of f_low marker
        x_ann = f_low + (f_high - f_low) * 0.08
        ax0.annotate(f'{db_low:.1f} dBFS',
                     xy=(f_low, db_low), xytext=(x_ann, db_low - 4),
                     color=GREEN, fontsize=10,
                     arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.0))
        ax0.annotate(f'{db_high:.1f} dBFS\nratio {ratio:+.1f} dB',
                     xy=(f_high, db_high),
                     xytext=(f_high * 1.03, db_high - 10),
                     color=AMBER, fontsize=10,
                     arrowprops=dict(arrowstyle='->', color=AMBER, lw=1.0))

        # Level check verdict inline on plot
        if 8 <= ratio <= 20:
            verdict, vcol = f'✓ flat input ({ratio:.1f} dB)', GREEN
        elif ratio > 28:
            verdict, vcol = f'⚠ RIAA detected! ({ratio:.1f} dB)', RED
        else:
            verdict, vcol = f'? check input ({ratio:.1f} dB)', AMBER
        ax0.text(0.02, 0.04, verdict, transform=ax0.transAxes,
                 color=vcol, fontsize=10, fontweight='bold',
                 va='bottom', ha='left')

        # ── Bottom: carrier zoom — FM sidebands ──────────────────────────
        ax1  = axes[1, col]
        ax1.set_facecolor(MID)
        mask2 = (freqs >= f_high - zoom_bw) & (freqs <= f_high + zoom_bw)
        ax1.plot(freqs[mask2], spec[mask2],
                 color=col_color, linewidth=0.9, alpha=0.92)
        ax1.axvline(f_high, color=WHITE, linewidth=1.0,
                    linestyle='--', alpha=0.6, label='Carrier')
        # Expected sideband positions
        for n in [1, 2, 3]:
            for sgn in [-1, +1]:
                sb = f_high + sgn * n * f_low
                if f_high - zoom_bw <= sb <= f_high + zoom_bw:
                    lbl = (f'±{n}×{f_low:.0f} Hz sidebands'
                           if n == 1 and sgn == 1 else '')
                    ax1.axvline(sb, color=GREEN, linewidth=0.8,
                                linestyle=':', alpha=0.7, label=lbl)
                    # label the sideband order at top of plot
                    ax1.text(sb, ax1.get_ylim()[1] if ax1.get_ylim()[1] != 0 else -10,
                             f'{n}×', color=GREEN, fontsize=9,
                             ha='center', va='bottom', alpha=0.8)
        ax1.set_xlabel('Frequency (Hz)', color=WHITE, fontsize=11)
        ax1.set_ylabel('Amplitude (dBFS)', color=WHITE, fontsize=11)
        ax1.legend(fontsize=10, facecolor=MID, edgecolor=MID, labelcolor=WHITE)
        _style_ax(ax1, f'Channel {ch}  —  Carrier zoom  (sidebands = FM deviation)')

        for ax in [ax0, ax1]:
            for sp in ax.spines.values():
                sp.set_color(WHITE)
            ax.tick_params(colors=WHITE, labelsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  SUMMARY REPORT  — averaged result across L and R channels
# ═════════════════════════════════════════════════════════════════════════════

def calibrate_peak_vel(wav_path: str,
                       known_vta_deg: float    = None,
                       f_low: float            = F_MOD,
                       f_high: float           = F_CARRIER,
                       theta_r: float          = RECORDING_ANGLE,
                       ref_displacement_um: float = 11.2,
                       ref_carrier_db: float      = -18.0,
                       calibration_method: str    = "both") -> dict:
    """
    Calculate PEAK_VEL using TWO independent methods and cross-check them.

    METHOD A — WAV amplitude ratio (requires linear recording chain):
        V_low = (amplitude_f_low / amplitude_f_high) × V_carrier_from_label
        V_carrier = 2π × f_high × displacement_f_high
        where displacement_f_high = ref_displacement_um × 10^(ref_carrier_db/20)
        Valid when preamp/soundcard gain is identical at f_low and f_high.

    METHOD B — Known cartridge VTA back-calculation:
        Run analyser with Method A result → measured VTA X
        PEAK_VEL_B = PEAK_VEL_A × tan(X°) / tan(known_vta°)
        Requires a cartridge with independently known VTA.
        Corrects for any remaining gain non-linearity.

    If known_vta_deg is None, only Method A is run.
    If both methods agree within 5%, the result is reliable.

    Args:
        wav_path             : WAV file of the test record band
        known_vta_deg        : cartridge true VTA in degrees (or None)
        f_low, f_high        : tone frequencies (Hz)
        theta_r              : recording angle (degrees)
        ref_displacement_um  : reference displacement from record label (µm)
        ref_carrier_db       : carrier level re reference (dB) — from label

    Returns:
        dict with keys: peak_vel_A, peak_vel_B (if known_vta given),
                        recommended, agreement_pct, per_band
    """
    if not _HAS_SF:
        raise ImportError("soundfile not installed — run: pip install soundfile")

    import soundfile as sf
    data, fs = sf.read(wav_path, always_2d=True)
    # Do NOT normalise — we need consistent amplitude levels
    # But do check for clipping
    peak_raw = float(np.max(np.abs(data)))
    data_norm = data / peak_raw   # normalise for analysis only

    W = 60   # box width

    print()
    print("╔" + "═"*W + "╗")
    print("║" + " PEAK_VEL CALIBRATION — DUAL METHOD ".center(W) + "║")
    print("╠" + "═"*W + "╣")
    print(f"║  WAV file    : {wav_path[-44:]:<44}  ║")
    print(f"║  Duration    : {len(data)/fs:.1f}s   Sample rate: {fs:.0f} Hz              ║")
    print(f"║  f_low       : {f_low:.0f} Hz    f_high: {f_high:.0f} Hz              ║")
    print(f"║  θR          : {theta_r:.1f}°    Ref disp: {ref_displacement_um:.1f} µm @ {ref_carrier_db:.0f} dB      ║")
    print("╠" + "═"*W + "╣")

    # ── Measure actual tone frequencies using segment means ──────────────
    # Wow causes the carrier to vary (e.g. 3934–3952 Hz) so the full-file
    # FFT peak locks onto the most common instantaneous frequency, not the
    # true mean. Per-segment peaks averaged give a better estimate of the
    # mean speed, which is what the label velocity refers to.
    audio_ref = data_norm[:, 0]
    N_full    = len(audio_ref)
    freqs_full = np.fft.rfftfreq(N_full, 1/fs)

    seg_len_cal = int(fs * 2.0)
    n_segs_cal  = max(3, N_full // seg_len_cal)

    f_low_segs  = []
    f_high_segs = []

    for i in range(n_segs_cal):
        sl  = slice(i * seg_len_cal, (i + 1) * seg_len_cal)
        seg = audio_ref[sl]
        if len(seg) < seg_len_cal // 2:
            continue
        freqs_s = np.fft.rfftfreq(len(seg), 1/fs)
        spec_s  = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) * 2 / len(seg)

        def seg_peak_freq(fc_nom, bw):
            m = (freqs_s >= fc_nom - bw) & (freqs_s <= fc_nom + bw)
            return float(freqs_s[m][np.argmax(spec_s[m])]) if np.any(m) else fc_nom

        f_low_segs.append( seg_peak_freq(f_low,  f_low  * 0.15))
        f_high_segs.append(seg_peak_freq(f_high, f_high * 0.10))

    f_low_actual  = float(np.mean(f_low_segs))   if f_low_segs  else f_low
    f_high_actual = float(np.mean(f_high_segs))  if f_high_segs else f_high
    f_high_min    = float(np.min(f_high_segs))   if f_high_segs else f_high
    f_high_max    = float(np.max(f_high_segs))   if f_high_segs else f_high
    wow_pp        = (f_high_max - f_high_min) / f_high * 100
    speed_error   = (f_high_actual - f_high) / f_high * 100

    print(f"║  Actual f_low  : {f_low_actual:.2f} Hz  (nominal {f_low:.0f} Hz, mean of {len(f_low_segs)} segments)  ║")
    print(f"║  Actual f_high : {f_high_actual:.2f} Hz  (nominal {f_high:.0f} Hz, mean of {len(f_high_segs)} segments)  ║")
    print(f"║  Speed error   : {speed_error:+.2f}%  (DC offset — turntable running slow/fast)  ║")
    print(f"║  Wow           : {wow_pp:.2f}% peak-to-peak  ({f_high_min:.1f}–{f_high_max:.1f} Hz range)  ║")
    print(f"║  Note: Mean speed used for V_carrier — corrects both speed error and wow  ║")

    # ── METHOD A: WAV amplitude ratio × label velocity ────────────────────
    print("╠" + "═"*W + "╣")
    print("║" + " METHOD A: WAV amplitude ratio × label velocity ".center(W) + "║")
    print("╠" + "═"*W + "╣")

    # Method A: only run when requested and label values are available.
    if calibration_method in ("A", "both") and ref_displacement_um > 0:
        disp_high_m = ref_displacement_um * 1e-6 * 10**(ref_carrier_db / 20)
        V_high      = 2 * np.pi * f_high_actual * disp_high_m   # speed-corrected

        print(f"║  Carrier displacement : {disp_high_m*1e9:.2f} nm                         ║")
        print(f"║  Carrier velocity     : {V_high*100:.4f} cm/s  (speed-corrected)     ║")

        amp_ratios = []
        for ch_idx, ch in enumerate(['L', 'R']):
            if ch_idx >= data.shape[1]:
                break
            audio_ch  = data_norm[:, ch_idx]
            N_ch      = len(audio_ch)
            freqs_ch  = np.fft.rfftfreq(N_ch, 1/fs)
            spec_ch   = np.abs(np.fft.rfft(audio_ch * np.hanning(N_ch))) * 2 / N_ch

            def pa(fc, bw):
                m = (freqs_ch >= fc - bw) & (freqs_ch <= fc + bw)
                return float(np.max(spec_ch[m])) if np.any(m) else 0.0

            a_low  = pa(f_low_actual,  bw=min(40, f_low_actual  * 0.08))
            a_high = pa(f_high_actual, bw=min(80, f_high_actual * 0.04))
            ratio  = a_low / a_high if a_high > 0 else 0.0
            amp_ratios.append(ratio)
            print(f"║  Ch {ch}: amp({f_low_actual:.0f}Hz)={a_low:.5f}  "
                  f"amp({f_high_actual:.0f}Hz)={a_high:.5f}  ratio={ratio:.4f} ║")

        ratio_mean  = float(np.mean(amp_ratios))
        peak_vel_A  = ratio_mean * V_high

        print(f"║  Mean amplitude ratio : {ratio_mean:.4f}  ({20*np.log10(ratio_mean):.1f} dB)        ║")
        print(f"║  ► METHOD A PEAK_VEL  : {peak_vel_A:.4f} m/s  ({peak_vel_A*100:.3f} cm/s)  ║")
    else:
        peak_vel_A = None
        print(f"║  Method A skipped — set CALIBRATION_METHOD='B' for non-CBS records ║")
        print(f"║  and set KNOWN_VTA to your cartridge's true VTA.                   ║")

    # ── METHOD B: known VTA back-calculation ──────────────────────────────
    peak_vel_B    = None
    agreement_pct = None

    if calibration_method in ("B", "both") and known_vta_deg is not None:
        print("╠" + "═"*W + "╣")
        print("║" + f" METHOD B: direct analytical solution (true VTA={known_vta_deg:.1f}°) ".center(W) + "║")
        print("╠" + "═"*W + "╣")

        # Direct analytical solution — no iteration needed.
        #
        # The VTA formula is:  θP = arctan(tan(θR) + C·R·F_signed / V_low)
        # Rearranging for V_low:
        #   V_low = C·R·F_signed / (tan(θP_known) - tan(θR))
        #
        # F_signed is measured from the signal and is INDEPENDENT of V_low
        # (it comes from the FM discriminator, not from the VTA formula).
        # So we can measure F with any trial V_low (we use V=1.0 as proxy),
        # then solve directly for the V_low that gives known_vta_deg.
        #
        # Because L and R channels measure slightly different F values
        # (real physical difference in cantilever geometry), we use mean |F|
        # to find the single PEAK_VEL where mean(VTA_L, VTA_R) = known_vta.
        # This matches the patent recommendation to average L and R.

        # C = 2π×rpm / (60×f_carrier×cos(θR))
        C_cal = 2.0*np.pi*TURNTABLE_RPM / (60.0*f_high*np.cos(np.radians(theta_r)))
        denom  = np.tan(np.radians(known_vta_deg)) - np.tan(np.radians(theta_r))

        if abs(denom) < 1e-6:
            print(f"║  ERROR: known_vta_deg equals theta_r — cannot calibrate     ║")
            peak_vel_B = None
        else:
            # Measure F_signed with trial V=1.0 (F is V-independent)
            F_vals = []
            for ch_idx, ch in enumerate(['L', 'R']):
                if ch_idx >= data.shape[1]:
                    break
                r_trial = analyse(data_norm[:, ch_idx], fs,
                                  channel=ch, f_low=f_low, f_high=f_high,
                                  R=GROOVE_RADIUS, V_low=1.0,
                                  theta_r=theta_r, verbose=False)
                F_vals.append(abs(r_trial['F_signed_hz']))
                print(f"║  Ch {ch} |F_signed| = {r_trial['F_signed_hz']:+8.3f} Hz                          ║")

            F_mean_cal = float(np.mean(F_vals))
            peak_vel_B = C_cal * GROOVE_RADIUS * F_mean_cal / denom

            print(f"║  Mean |F|            = {F_mean_cal:8.3f} Hz                          ║")
            print(f"║  Formula: V = C·R·F / (tan({known_vta_deg:.0f}°)-tan({theta_r:.0f}°))             ║")
            print(f"║         = {C_cal:.5f}·{GROOVE_RADIUS}·{F_mean_cal:.3f} / {denom:.6f}        ║")
            print(f"║  ► METHOD B PEAK_VEL  : {peak_vel_B:.4f} m/s  ({peak_vel_B*100:.3f} cm/s)  ║")

            # Verify — both channels should average to known_vta
            vtas_verify = []
            for ch_idx, ch in enumerate(['L', 'R']):
                if ch_idx >= data.shape[1]:
                    break
                r_v = analyse(data_norm[:, ch_idx], fs,
                              channel=ch, f_low=f_low, f_high=f_high,
                              R=GROOVE_RADIUS, V_low=peak_vel_B,
                              theta_r=theta_r, verbose=False)
                vtas_verify.append(r_v['vta_deg'])
                print(f"║  Verify Ch {ch}: VTA={r_v['vta_deg']:6.3f}°                              ║")
            print(f"║  Verify mean:  VTA={np.mean(vtas_verify):6.3f}°  (target {known_vta_deg:.1f}°)             ║")

        # Agreement between methods
        agreement_pct = (abs(peak_vel_A - peak_vel_B) / peak_vel_B * 100
                        if peak_vel_A is not None else None)
        print("╠" + "═"*W + "╣")
        if agreement_pct is not None:
            print(f"║  Agreement A vs B     : {agreement_pct:.1f}%                              ║")
            if agreement_pct < 3:
                verdict = "EXCELLENT — chain is linear, both methods consistent"
            elif agreement_pct < 7:
                verdict = "GOOD — minor chain non-linearity, use Method B"
            else:
                verdict = "POOR — significant chain non-linearity, use Method B"
            print(f"║  {verdict:<58}║")
        else:
            print(f"║  Agreement A vs B     : N/A (Method A not available)          ║")
            print(f"║  Use Method B PEAK_VEL value directly.                        ║")

    # ── Recommended value ─────────────────────────────────────────────────
    recommended = peak_vel_B if peak_vel_B is not None else peak_vel_A
    method_used = "B (VTA back-calc)" if peak_vel_B is not None else "A (WAV ratio)"

    print("╠" + "═"*W + "╣")
    print("║" + " RECOMMENDED PEAK_VEL VALUES ".center(W) + "║")
    print("╠" + "═"*W + "╣")
    print(f"║  Using Method {method_used:<46}║")
    print(f"║  This band ({f_low:.0f}Hz)                                        ║")
    for db_offset, db_label in [(0, '+6dB (this band)'),
                                 (3, '+9dB            '),
                                 (6, '+12dB           ')]:
        v = recommended * 10**(db_offset / 20)
        print(f"║    {f_low:.0f}Hz {db_label} : {v:.4f} m/s  ({v*100:.3f} cm/s)        ║")

    # Half-frequency companion bands if f_low == 400
    if abs(f_low - 400) < 1:
        print(f"║  200Hz companion bands (÷2, same groove depth):            ║")
        for db_offset, db_label in [(0, '+6dB (this band)'),
                                     (3, '+9dB            '),
                                     (6, '+12dB           ')]:
            v = recommended * 0.5 * 10**(db_offset / 20)
            print(f"║    200Hz {db_label} : {v:.4f} m/s  ({v*100:.3f} cm/s)        ║")

    print("╚" + "═"*W + "╝")
    print()
    print(f">>> Copy into settings:  PEAK_VEL = {recommended:.4f}")

    return dict(peak_vel_A      = peak_vel_A,
                peak_vel_B      = peak_vel_B,
                recommended     = recommended,
                agreement_pct   = agreement_pct,
                f_low_actual    = f_low_actual,
                f_high_actual   = f_high_actual,
                f_high_min      = f_high_min,
                f_high_max      = f_high_max,
                wow_pp_pct      = wow_pp,
                speed_error_pct = speed_error)


def print_summary(results_list: list):
    """
    Print a final summary. If both L and R channels were measured,
    average them as the patent recommends (Col. 7):
    'it is acceptable to average the vertical angle readings obtained
    on the left and right channels to obtain the final measurement.'
    Also give a tonearm adjustment recommendation.
    """
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║              FINAL SUMMARY REPORT                   ║")
    print("╠══════════════════════════════════════════════════════╣")

    vta_values = [r['vta_deg'] for r in results_list]
    theta_r    = results_list[0]['theta_r_deg']

    for r in results_list:
        _lbl = "HTA" if r.get('modulation','vertical') == 'lateral' else "VTA"
        print(f"║  Channel {r['channel']}: {_lbl} = {r['vta_deg']:6.2f}°  "
              f"(error {r['vta_error_deg']:+.2f}°)         ║")

    if len(vta_values) == 2:
        vta_avg = float(np.mean(vta_values))
        err_avg = vta_avg - theta_r
        print(f"╠══════════════════════════════════════════════════════╣")
        _avglbl = "HTA" if any(r.get('modulation','vertical')=='lateral' for r in results_list) else "VTA"
        print(f"║  Average {_avglbl} : {vta_avg:6.2f}°  (error {err_avg:+.2f}°)            ║")
    else:
        err_avg = results_list[0]['vta_error_deg']
        vta_avg = results_list[0]['vta_deg']

    print(f"╠══════════════════════════════════════════════════════╣")
    _reclbl = "Lateral rec angle θR" if any(r.get('modulation','vertical')=='lateral' for r in results_list) else "Vertical rec angle θR"
    print(f"║  {_reclbl} : {theta_r:.1f}°                          ║")
    _mvlbl = "HTA" if any(r.get('modulation','vertical')=='lateral' for r in results_list) else "VTA"
    print(f"║  Measured {_mvlbl}       : {vta_avg:.2f}°                         ║")

    abs_err = abs(err_avg)
    if abs_err < 0.5:
        verdict = "EXCELLENT — VTA is within ±0.5° of recording angle"
        action  = "No adjustment needed."
    elif abs_err < 2.0:
        verdict = "GOOD      — VTA is within ±2°"
        action  = "Minor adjustment beneficial but optional."
    elif abs_err < 5.0:
        verdict = "FAIR      — VTA error is significant"
        action  = "Adjustment recommended."
    else:
        verdict = "POOR      — VTA error is large, distortion likely"
        action  = "Adjustment required."

    print(f"╠══════════════════════════════════════════════════════╣")
    print(f"║  {verdict[:52]:<52}║")
    print(f"╠══════════════════════════════════════════════════════╣")

    _adj_lbl = "HTA" if any(r.get('modulation','vertical')=='lateral' for r in results_list) else "VTA"
    if err_avg > 0.5:
        print(f"║  ADJUSTMENT: {_adj_lbl} too STEEP by {err_avg:+.1f}°                  ║")
        print(f"║  → Lower the rear of the tonearm (or raise front)    ║")
        print(f"║    until meter reads {theta_r:.1f}°                          ║")
    elif err_avg < -0.5:
        print(f"║  ADJUSTMENT: {_adj_lbl} too SHALLOW by {abs(err_avg):.1f}°               ║")
        print(f"║  → Raise the rear of the tonearm (or lower front)    ║")
        print(f"║    until meter reads {theta_r:.1f}°                          ║")
    else:
        print(f"║  {action:<52}║")

    # Tracing distortion summary
    print(f"╠══════════════════════════════════════════════════════╣")
    print(f"║  Tracing distortion (2nd harmonic):                  ║")
    for r in results_list:
        h2 = r['harmonics_db'].get(2, -120)
        td_verdict = ("low" if h2 < -40 else
                      "moderate" if h2 < -20 else "HIGH — check stylus condition")
        print(f"║    Ch {r['channel']}: {h2:+6.1f} dB re fundamental  ({td_verdict:<20})║")

    print(f"╚══════════════════════════════════════════════════════╝")


# ═════════════════════════════════════════════════════════════════════════════
#  SPYDER / INTERACTIVE ENTRY POINT
#  Edit the settings below then press F5 (Run file) in Spyder
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Self-test: verify sign handling is correct ──────────────────
    _audio_test = generate_test_signal(fs=44100., vta_error_deg=5.,
                                        channel='L', f_low=400., f_high=4000.)
    _r = analyse(_audio_test, 44100., channel='L', f_low=400., f_high=4000.,
                 R=0.114, V_low=0.0563, theta_r=16.0, verbose=False)
    _vta_ok = 18.0 < _r['vta_deg'] < 24.0   # should be ~21 deg
    if _vta_ok:
        print('VTA Analyzer v2.0 — self-test PASSED ✓  Left channel sign is correct')
    else:
        print(f'*** WARNING: self-test FAILED — Left ch VTA={_r["vta_deg"]:.1f}° (expected ~21°)')
        print('*** Old code may be cached. Close Spyder completely, reopen, then F5.')
    print()

    # Settings are defined at the TOP of this file — scroll up to edit them.
    # (Look for the "USER SETTINGS" banner near line 70)

    results_all = []

    # ── Settings summary — printed every run to catch entry errors ────────────
    W = 47
    mode_label = "MODE 1 — CBS STR-112" if TONE_PAIR != 'custom' else "MODE 2 — CUSTOM"
    ang_label  = "HTA" if MODULATION == 'lateral' else "VTA"
    print()
    print("╔" + "═"*W + "╗")
    print("║" + f"  {mode_label}".ljust(W) + "║")
    print("╠" + "═"*W + "╣")
    if TONE_PAIR != 'custom':
        print("║" + f"  Tone pair   : {TONE_PAIR}".ljust(W) + "║")
        print("║" + f"  Modulation  : {MODULATION}  ({ang_label})".ljust(W) + "║")
        print("║" + f"  Band        : {BAND}".ljust(W) + "║")
        print("╠" + "═"*W + "╣")
    else:
        print("║" + f"  Modulation  : {MODULATION}  ({ang_label})".ljust(W) + "║")
        print("╠" + "═"*W + "╣")
    import math as _math
    _C = 2*_math.pi*TURNTABLE_RPM / (60*F_CARRIER*_math.cos(_math.radians(RECORDING_ANGLE)))
    print("║" + f"  → F_MOD     = {F_MOD:.0f} Hz".ljust(W) + "║")
    print("║" + f"  → F_CARRIER = {F_CARRIER:.0f} Hz".ljust(W) + "║")
    print("║" + f"  → PEAK_VEL  = {PEAK_VEL:.4f} m/s".ljust(W) + "║")
    print("║" + f"  → GROOVE_R  = {GROOVE_RADIUS:.3f} m  ({GROOVE_RADIUS*1000:.0f} mm)".ljust(W) + "║")
    print("║" + f"  → REC_ANGLE = {RECORDING_ANGLE:.1f}°".ljust(W) + "║")
    print("║" + f"  → COEFF C   = {_C:.4e}".ljust(W) + "║")
    print("╚" + "═"*W + "╝")
    print()


    # ── Load or generate audio ────────────────────────────────────────────
    if DEMO_MODE:
        print(f"\n[DEMO MODE] Synthetic CBS STR-112 signal")
        print(f"  f_low={F_MOD:.0f} Hz, injected VTA error = {DEMO_VTA_ERROR:+.1f}°  "
              f"(θP = {RECORDING_ANGLE + DEMO_VTA_ERROR:.1f}°)")
        channels  = ['L', 'R'] if CHANNEL == 'both' else [CHANNEL]
        fs        = 44100.0
        audio_map = {ch: generate_test_signal(fs=fs,
                                               vta_error_deg=DEMO_VTA_ERROR,
                                               channel=ch,
                                               f_low=F_MOD,
                                               f_high=F_CARRIER)
                     for ch in channels}
    else:
        if not _HAS_SF:
            raise ImportError(
                "soundfile is not installed.\n"
                "In the Spyder IPython console run:  pip install soundfile\n"
                "Then restart the kernel (circular arrow button).")
        import soundfile as sf
        data, fs = sf.read(AUDIO_FILE, always_2d=True)
        pk = np.max(np.abs(data))
        if pk > 0:
            data = data / pk

        # Inverse RIAA correction removed — FM method is RIAA-immune
            # Re-normalise after filtering
            pk2 = np.max(np.abs(data))
            if pk2 > 0:
                data = data / pk2


        print(f"\nLoaded: {AUDIO_FILE}")
        print(f"  Sample rate : {fs:.0f} Hz")
        print(f"  Duration    : {data.shape[0]/fs:.1f} s")
        print(f"  Channels    : {data.shape[1]}")
        if CHANNEL == 'both' and data.shape[1] >= 2:
            channels = ['L', 'R']
        else:
            channels = [CHANNEL if CHANNEL != 'both' else 'L']
        audio_map = {
            ch: data[:, (0 if ch == 'L' else min(1, data.shape[1]-1))]
            for ch in channels
        }

    # ── Spectrum check (before analysis — verify tones are present) ───────
    if SHOW_SPECTRUM:
        fig_spec = plot_spectrum(audio_map, fs,
                                 f_low=F_MOD, f_high=F_CARRIER)
        plt.show(block=False)

    # ── Calibration mode ──────────────────────────────────────────────────
    if CALIBRATION_MODE and not DEMO_MODE:
        calibrate_peak_vel(
            wav_path             = AUDIO_FILE,
            known_vta_deg        = KNOWN_VTA,
            f_low                = F_MOD,
            f_high               = F_CARRIER,
            theta_r              = RECORDING_ANGLE,
            ref_displacement_um  = REF_DISPLACEMENT_UM,
            ref_carrier_db       = REF_CARRIER_DB,
            calibration_method   = CALIBRATION_METHOD)
        print("Calibration complete. Update PEAK_VEL in settings, then set")
        print("CALIBRATION_MODE = False and run again for normal measurement.")
        import sys; sys.exit(0)

    # ── Run analysis on each channel ──────────────────────────────────────
    # Resolve effective recording angle and groove radius for this run
    _eff_theta_r = RECORDING_ANGLE   # already 0.0 if lateral band was auto-set
    _eff_R       = GROOVE_RADIUS      # already set from BAND lookup above
    _eff_V       = PEAK_VEL           # already set from BAND lookup above

    for ch, audio in audio_map.items():
        res = analyse(audio, fs,
                      channel        = ch,
                      f_low          = F_MOD,
                      f_high         = F_CARRIER,
                      R              = _eff_R,
                      V_low          = _eff_V,
                      theta_r        = _eff_theta_r,
                      swap_channels  = SWAP_CHANNELS,
                      modulation     = MODULATION,
                      verbose        = True)
        results_all.append(res)

    # ── Summary report ────────────────────────────────────────────────────
    print_summary(results_all)

    # ── Signal chain and meter plots ──────────────────────────────────────
    if SHOW_SIGNAL_CHAIN:
        fig_chain = plot_signal_chain(results_all[0])
        plt.show(block=False)

    if SHOW_METER:
        fig_meter = plot_meter_dashboard(results_all,
                                         script_name  = __file__,
                                         wav_name     = (AUDIO_FILE if not DEMO_MODE else ''),
                                         description  = METER_DESCRIPTION)
        plt.show(block=False)

    plt.show()  # Block here so all windows stay open
