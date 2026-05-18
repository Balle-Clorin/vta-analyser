# -*- coding: utf-8 -*-
"""
VTA Analyzer — Streamlit Web App
Wraps vta_analyzer_core.py with a browser-based UI.

Run with:
    streamlit run app.py
"""

import io
import os
import sys
import warnings
import tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless — no display needed
import matplotlib.pyplot as plt
import streamlit as st

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="VTA Analyzer",
    page_icon="💿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark vinyl-lab theme */
    .stApp { background-color: #0e1117; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #e8eaf0; }
    .result-card {
        background: #1c2230;
        border: 1px solid #2d3548;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-big {
        font-size: 2.4rem;
        font-weight: 700;
        font-family: monospace;
    }
    .green  { color: #39ff14; }
    .amber  { color: #ffb300; }
    .red    { color: #ff3d3d; }
    .accent { color: #00d4ff; }
    .muted  { color: #8892a4; font-size: 0.85rem; }
    .banner {
        background: #1c2230;
        border-left: 4px solid #00d4ff;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        font-family: monospace;
        font-size: 0.88rem;
        color: #b0bcd0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load the core module (import as module, patching globals) ─────────────────
@st.cache_resource
def load_core():
    """Import the core analyser module once and return it."""
    import importlib.util, types

    path = os.path.join(os.path.dirname(__file__), "vta_analyzer_core.py")
    spec = importlib.util.spec_from_file_location("vta_core", path)
    mod  = importlib.util.module_from_spec(spec)
    # Suppress the module-level self-test printout
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass
        finally:
            sys.stdout = old_stdout
    return mod

core = load_core()

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR — all settings
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💿 VTA Analyzer")
    st.markdown('<div class="muted">CBS STR-112 & custom records<br>US Patent 4,359,768</div>',
                unsafe_allow_html=True)
    st.divider()

    # ── Record mode ──────────────────────────────────────────────────────────
    mode = st.radio("Record mode", ["CBS STR-112 (auto)", "Custom record"],
                    horizontal=False)

    st.divider()

    if mode == "CBS STR-112 (auto)":
        tone_pair  = st.selectbox("Tone pair",  ["400+4000", "200+4000"])
        modulation = st.selectbox("Modulation", ["vertical", "lateral"])

        # Available bands depend on modulation
        if modulation == "lateral":
            band_opts = ["+6dB", "+9dB", "+12dB", "+15dB"]
        else:
            band_opts = ["+6dB", "+9dB", "+12dB"]
        band = st.selectbox("Band level", band_opts)

    else:  # Custom
        modulation   = st.selectbox("Modulation", ["vertical", "lateral"])
        f_mod_custom = st.number_input("Modulating frequency (Hz)", value=400.0,
                                       min_value=10.0, max_value=2000.0, step=10.0)
        f_car_custom = st.number_input("Carrier frequency (Hz)",    value=4000.0,
                                       min_value=500.0, max_value=20000.0, step=100.0)
        peak_vel_c   = st.number_input("Peak velocity (m/s)",       value=0.0563,
                                       min_value=0.001, max_value=1.0, step=0.001,
                                       format="%.4f")
        groove_r_c   = st.number_input("Groove radius (m)",         value=0.114,
                                       min_value=0.010, max_value=0.200, step=0.001,
                                       format="%.3f")
        rec_angle_c  = st.number_input("Recording angle (°)",       value=16.5,
                                       min_value=0.0, max_value=30.0, step=0.1)

    st.divider()

    channel       = st.selectbox("Channel(s)", ["both", "L", "R"])
    swap_channels = st.checkbox("Swap L/R channels", value=False)
    rpm           = st.number_input("Turntable speed (RPM)", value=33.333,
                                    min_value=16.0, max_value=78.0, step=0.001,
                                    format="%.3f")
    apply_riaa    = st.checkbox("Apply inverse RIAA correction", value=False)

    st.divider()

    # ── Optional displays ────────────────────────────────────────────────────
    show_spectrum = st.checkbox("Show spectrum plot",      value=False)
    show_chain    = st.checkbox("Show signal chain plot",  value=False)

    st.divider()
    demo_mode     = st.checkbox("Demo mode (no WAV needed)", value=False)
    if demo_mode:
        demo_error = st.slider("Injected VTA error (°)", -10.0, 10.0, 5.0, 0.5)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PANEL
# ─────────────────────────────────────────────────────────────────────────────
st.title("💿 Vertical Tracking Angle Analyzer")
st.markdown(
    '<div class="banner">'
    'Software implementation of CBS VTA Meter &nbsp;·&nbsp; US Patent 4,359,768 '
    '(Abbagnaro & Gust, 1982)'
    '</div>',
    unsafe_allow_html=True,
)

# ── File upload (or demo) ────────────────────────────────────────────────────
if not demo_mode:
    uploaded = st.file_uploader(
        "Upload your WAV recording of the test record",
        type=["wav"],
        help="Record the test band flat (no RIAA), or enable 'Apply inverse RIAA' in the sidebar.",
    )
else:
    uploaded = None
    st.info("🔬 Demo mode — synthetic CBS STR-112 signal will be generated.")

run_btn = st.button("▶  Analyse", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
#  RUN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    if not demo_mode and uploaded is None:
        st.error("Please upload a WAV file, or enable Demo mode in the sidebar.")
        st.stop()

    # ── Patch core module globals with sidebar settings ───────────────────
    if mode == "CBS STR-112 (auto)":
        # Normalise band string: "+6dB" → "+6dB" (already correct)
        band_key = band   # e.g. "+6dB"

        core.TONE_PAIR   = tone_pair
        core.MODULATION  = modulation
        core.BAND        = band_key

        _band_table = (core._BANDS if modulation == 'lateral'
                       else core._VERTICAL_BANDS)
        _tone_table = _band_table.get(tone_pair, {})
        core.PEAK_VEL      = _tone_table[band_key]['peak_vel']
        core.GROOVE_RADIUS = _tone_table[band_key]['groove_radius']
        core.RECORDING_ANGLE = 0.0 if modulation == 'lateral' else 16.5
        core.F_MOD     = 200 if tone_pair == '200+4000' else 400
        core.F_CARRIER = 4000

    else:  # Custom
        core.MODULATION      = modulation
        core.F_MOD           = f_mod_custom
        core.F_CARRIER       = f_car_custom
        core.PEAK_VEL        = peak_vel_c
        core.GROOVE_RADIUS   = groove_r_c
        core.RECORDING_ANGLE = rec_angle_c
        core.TONE_PAIR       = 'custom'
        core.BAND            = 'custom'

    core.CHANNEL        = channel
    core.SWAP_CHANNELS  = swap_channels
    core.TURNTABLE_RPM  = rpm

    # ── Load / generate audio ─────────────────────────────────────────────
    with st.spinner("Loading audio…"):
        try:
            import soundfile as sf
        except ImportError:
            st.error(
                "The `soundfile` library is not installed.\n\n"
                "Open a terminal and run:\n```\npip install soundfile\n```\n"
                "Then restart the app."
            )
            st.stop()

        channels_to_run = ['L', 'R'] if channel == 'both' else [channel]

        if demo_mode:
            fs = 44100.0
            audio_map = {
                ch: core.generate_test_signal(
                    fs=fs,
                    vta_error_deg=demo_error,
                    channel=ch,
                    f_low=core.F_MOD,
                    f_high=core.F_CARRIER,
                )
                for ch in channels_to_run
            }
            wav_name = "(demo/synthetic)"
        else:
            raw = uploaded.read()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name

            data, fs = sf.read(tmp_path, always_2d=True)
            os.unlink(tmp_path)

            pk = np.max(np.abs(data))
            if pk > 0:
                data = data / pk

            if apply_riaa:
                data = core.apply_inverse_riaa(data, fs)
                pk2  = np.max(np.abs(data))
                if pk2 > 0:
                    data = data / pk2

            if channel == 'both' and data.shape[1] >= 2:
                channels_to_run = ['L', 'R']
            else:
                channels_to_run = [channel if channel != 'both' else 'L']

            audio_map = {
                ch: data[:, (0 if ch == 'L' else min(1, data.shape[1]-1))]
                for ch in channels_to_run
            }
            wav_name = uploaded.name

    # ── Run analysis ──────────────────────────────────────────────────────
    results_all = []
    progress    = st.progress(0, text="Analysing…")

    for i, (ch, audio) in enumerate(audio_map.items()):
        progress.progress((i) / len(audio_map), text=f"Analysing channel {ch}…")

        captured = io.StringIO()
        old_out  = sys.stdout
        sys.stdout = captured

        with warnings.catch_warnings(record=True) as caught_warns:
            warnings.simplefilter("always")
            res = core.analyse(
                audio, fs,
                channel       = ch,
                f_low         = core.F_MOD,
                f_high        = core.F_CARRIER,
                R             = core.GROOVE_RADIUS,
                V_low         = core.PEAK_VEL,
                theta_r       = core.RECORDING_ANGLE,
                swap_channels = swap_channels,
                modulation    = core.MODULATION,
                verbose       = True,
            )

        sys.stdout = old_out
        results_all.append(res)

        # Surface any RIAA warnings
        for w in caught_warns:
            if issubclass(w.category, UserWarning):
                st.warning(str(w.message))

    progress.progress(1.0, text="Done ✓")

    # ─────────────────────────────────────────────────────────────────────
    #  RESULTS DISPLAY
    # ─────────────────────────────────────────────────────────────────────
    ang_label = "HTA" if core.MODULATION == 'lateral' else "VTA"

    st.divider()
    st.subheader(f"📊 {ang_label} Results — {wav_name}")

    # ── Average badge ─────────────────────────────────────────────────────
    if len(results_all) >= 2:
        vta_vals = [r['vta_deg'] for r in results_all]
        vta_avg  = float(np.mean(vta_vals))
        err_avg  = vta_avg - results_all[0]['theta_r_deg']
        avg_cls  = "green" if abs(err_avg) < 2 else "amber" if abs(err_avg) < 5 else "red"
        st.markdown(
            f'<div style="text-align:center; margin-bottom:1rem;">'
            f'<span class="metric-big {avg_cls}">'
            f'AVG {ang_label}: {vta_avg:.2f}°&nbsp;&nbsp;'
            f'<span style="font-size:1.4rem">(err {err_avg:+.2f}°)</span>'
            f'</span></div>',
            unsafe_allow_html=True,
        )

    # ── Per-channel metric cards ──────────────────────────────────────────
    cols = st.columns(len(results_all))
    for col, res in zip(cols, results_all):
        err   = res['vta_error_deg']
        cls   = "green" if abs(err) < 2 else "amber" if abs(err) < 5 else "red"
        h2_db = res['harmonics_db'].get(2, -120)
        td    = ("✅ Low" if h2_db < -40 else
                 "⚠️ Moderate" if h2_db < -20 else "❌ HIGH")
        with col:
            st.markdown(f"""
<div class="result-card">
  <div class="muted">Channel {res['channel']}</div>
  <div class="metric-big {cls}">{res['vta_deg']:.2f}°</div>
  <div class="muted">{ang_label} &nbsp;·&nbsp; err <span class="{cls}">{err:+.2f}°</span></div>
  <hr style="border-color:#2d3548; margin:0.6rem 0">
  <div class="muted">Peak FM deviation: <span class="accent">{res['F_peak_hz']:.1f} Hz</span></div>
  <div class="muted">Phase φ: <span class="accent">{res['phi_deg']:+.1f}°</span></div>
  <div class="muted">DC tracking: <span class="accent">{res['dc_tracking']:+.4f}</span></div>
  <div class="muted">DC tracing:  <span class="accent">{res['dc_tracing']:+.4f}</span></div>
  <hr style="border-color:#2d3548; margin:0.6rem 0">
  <div class="muted">2nd harmonic: <span class="accent">{h2_db:+.1f} dB</span></div>
  <div class="muted">Tracing distortion: {td}</div>
</div>
""", unsafe_allow_html=True)

    # ── Tonearm adjustment advice ─────────────────────────────────────────
    theta_r = results_all[0]['theta_r_deg']
    avg_err  = float(np.mean([r['vta_error_deg'] for r in results_all]))
    if abs(avg_err) < 1.0:
        advice_col, advice = "green",  f"✅ VTA is within ±1° of θR={theta_r:.1f}° — no adjustment needed."
    elif avg_err > 0:
        advice_col, advice = "amber", (f"⬆️  VTA reads {avg_err:+.1f}° too steep. "
                                       f"Lower the rear of the tonearm (or raise the front).")
    else:
        advice_col, advice = "amber", (f"⬇️  VTA reads {avg_err:+.1f}° too shallow. "
                                       f"Raise the rear of the tonearm (or lower the front).")

    st.markdown(
        f'<div class="result-card"><span class="{advice_col}">{advice}</span></div>',
        unsafe_allow_html=True,
    )

    # ── Meter dashboard figure ─────────────────────────────────────────────
    st.subheader("🎛️ Meter Dashboard")
    fig_meter = core.plot_meter_dashboard(
        results_all,
        script_name  = "vta_analyzer_core.py",
        wav_name     = wav_name,
        description  = "",
    )
    st.pyplot(fig_meter, use_container_width=True)
    plt.close(fig_meter)

    # ── Optional: spectrum ─────────────────────────────────────────────────
    if show_spectrum:
        st.subheader("📈 Spectrum")
        fig_spec = core.plot_spectrum(
            audio_map, fs,
            f_low=core.F_MOD, f_high=core.F_CARRIER,
        )
        st.pyplot(fig_spec, use_container_width=True)
        plt.close(fig_spec)

    # ── Optional: signal chain ─────────────────────────────────────────────
    if show_chain:
        st.subheader("🔗 Patent Signal Chain")
        fig_chain = core.plot_signal_chain(results_all[0])
        st.pyplot(fig_chain, use_container_width=True)
        plt.close(fig_chain)

    # ── Settings summary ──────────────────────────────────────────────────
    with st.expander("⚙️ Settings used for this run"):
        r0 = results_all[0]
        st.code(
            f"Mode        : {mode}\n"
            f"Tone pair   : {getattr(core,'TONE_PAIR','custom')}\n"
            f"Modulation  : {core.MODULATION}  ({ang_label})\n"
            f"Band        : {getattr(core,'BAND','custom')}\n"
            f"F_MOD       : {core.F_MOD:.0f} Hz\n"
            f"F_CARRIER   : {core.F_CARRIER:.0f} Hz\n"
            f"PEAK_VEL    : {core.PEAK_VEL:.4f} m/s\n"
            f"GROOVE_R    : {core.GROOVE_RADIUS:.3f} m  ({core.GROOVE_RADIUS*1000:.0f} mm)\n"
            f"REC_ANGLE   : {core.RECORDING_ANGLE:.1f}°\n"
            f"RPM         : {rpm:.3f}\n"
            f"Channel     : {channel}\n"
            f"Swap L/R    : {swap_channels}\n"
            f"Inv RIAA    : {apply_riaa}\n"
            f"Sample rate : {r0['fs']:.0f} Hz",
            language="text",
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div class="muted" style="text-align:center">'
    'VTA Analyzer &nbsp;·&nbsp; Based on White &amp; Gust (1979) &amp; US Patent 4,359,768 &nbsp;·&nbsp; '
    'Built with Streamlit</div>',
    unsafe_allow_html=True,
)
