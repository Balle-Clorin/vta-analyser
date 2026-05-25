# -*- coding: utf-8 -*-
"""
VTA Analyzer — Streamlit Web App
Wraps vta_analyzer_core.py with a browser-based UI.

Run with:
    streamlit run app.py
"""

import gc
import io
import os
import sys
import warnings
import tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VTA Analyzer",
    page_icon="💿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
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
    .cal-card {
        background: #1a1f2e;
        border: 2px solid #ffb300;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-big { font-size: 2.4rem; font-weight: 700; font-family: monospace; }
    .metric-med { font-size: 1.6rem; font-weight: 700; font-family: monospace; }
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
    .cal-banner {
        background: #1c2230;
        border-left: 4px solid #ffb300;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        font-family: monospace;
        font-size: 0.88rem;
        color: #b0bcd0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load the core module ──────────────────────────────────────────────────────
@st.cache_resource
def load_core():
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "vta_analyzer_core.py")
    spec = importlib.util.spec_from_file_location("vta_core", path)
    mod  = importlib.util.module_from_spec(spec)
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

# ── Memory cleanup helper ─────────────────────────────────────────────────────
def cleanup_memory(*arrays):
    """Delete numpy arrays and force garbage collection."""
    for arr in arrays:
        try:
            del arr
        except Exception:
            pass
    plt.close('all')   # close any stray matplotlib figures
    gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💿 VTA Analyzer")
    st.markdown('<div class="muted">CBS STR-112 & custom records<br>US Patent 4,359,768</div>',
                unsafe_allow_html=True)
    st.divider()

    # ── App mode (Analyse vs Calibrate) ──────────────────────────────────────
    app_mode = st.radio("App mode", ["📊 Analyse", "🔧 Calibrate PEAK_VEL"],
                        horizontal=False)

    st.divider()

    # ── Record mode ───────────────────────────────────────────────────────────
    mode = st.radio("Record mode", ["CBS STR-112 (auto)", "Custom record"],
                    horizontal=False)

    st.divider()

    if mode == "CBS STR-112 (auto)":
        tone_pair  = st.selectbox("Tone pair",  ["400+4000", "200+4000"])
        modulation = st.selectbox("Modulation", ["vertical", "lateral"])
        if modulation == "lateral":
            band_opts = ["+6dB", "+9dB", "+12dB", "+15dB"]
        else:
            band_opts = ["+6dB", "+9dB", "+12dB"]
        band = st.selectbox("Band level", band_opts)
    else:
        modulation   = st.selectbox("Modulation", ["vertical", "lateral"])
        f_mod_custom = st.number_input("Modulating frequency (Hz)", value=400.0,
                                       min_value=10.0, max_value=2000.0, step=10.0)
        f_car_custom = st.number_input("Carrier frequency (Hz)", value=4000.0,
                                       min_value=500.0, max_value=20000.0, step=100.0)
        peak_vel_c   = st.number_input("Peak velocity (m/s)", value=0.0563,
                                       min_value=0.001, max_value=1.0, step=0.001,
                                       format="%.4f")
        groove_r_c   = st.number_input("Groove radius (m)", value=0.114,
                                       min_value=0.010, max_value=0.200, step=0.001,
                                       format="%.3f")
        rec_angle_c  = st.number_input("Recording angle (°)", value=16.5,
                                       min_value=0.0, max_value=30.0, step=0.1)

    st.divider()

    channel       = st.selectbox("Channel(s)", ["both", "L", "R"])
    swap_channels = st.checkbox("Swap L/R channels", value=False)
    rpm           = st.number_input("Turntable speed (RPM)", value=33.333,
                                    min_value=16.0, max_value=78.0, step=0.001,
                                    format="%.3f")
    apply_riaa    = st.checkbox("Apply inverse RIAA correction", value=False)

    st.divider()

    show_spectrum = st.checkbox("Show spectrum plot",     value=False)
    show_chain    = st.checkbox("Show signal chain plot", value=False)

    st.divider()
    demo_mode = st.checkbox("Demo mode (no WAV needed)", value=False)
    if demo_mode:
        demo_error = st.slider("Injected VTA error (°)", -10.0, 10.0, 5.0, 0.5)

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: apply sidebar settings to core globals
# ─────────────────────────────────────────────────────────────────────────────
def apply_settings():
    core.TURNTABLE_RPM  = rpm
    core.SWAP_CHANNELS  = swap_channels
    core.CHANNEL        = channel

    if mode == "CBS STR-112 (auto)":
        band_key = band
        core.TONE_PAIR       = tone_pair
        core.MODULATION      = modulation
        core.BAND            = band_key
        _band_table          = (core._BANDS if modulation == 'lateral'
                                else core._VERTICAL_BANDS)
        _tone_table          = _band_table.get(tone_pair, {})
        core.PEAK_VEL        = _tone_table[band_key]['peak_vel']
        core.GROOVE_RADIUS   = _tone_table[band_key]['groove_radius']
        core.RECORDING_ANGLE = 0.0 if modulation == 'lateral' else 16.5
        core.F_MOD           = 200 if tone_pair == '200+4000' else 400
        core.F_CARRIER       = 4000
    else:
        core.TONE_PAIR       = 'custom'
        core.BAND            = 'custom'
        core.MODULATION      = modulation
        core.F_MOD           = f_mod_custom
        core.F_CARRIER       = f_car_custom
        core.PEAK_VEL        = peak_vel_c
        core.GROOVE_RADIUS   = groove_r_c
        core.RECORDING_ANGLE = rec_angle_c

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: load WAV from upload → numpy array + fs
# ─────────────────────────────────────────────────────────────────────────────
def load_wav(uploaded_file):
    try:
        import soundfile as sf
    except ImportError:
        st.error("The `soundfile` library is not installed.\n\n"
                 "Run: `pip install soundfile` then restart the app.")
        st.stop()

    raw = uploaded_file.read()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        del raw   # free the raw bytes immediately
        gc.collect()

        data, fs = sf.read(tmp_path, always_2d=True)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    pk = np.max(np.abs(data))
    if pk > 0:
        data = data / pk

    if apply_riaa:
        data = core.apply_inverse_riaa(data, fs)
        pk2  = np.max(np.abs(data))
        if pk2 > 0:
            data = data / pk2

    return data, fs


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN PANEL — title always visible
# ═════════════════════════════════════════════════════════════════════════════
st.title("💿 Vertical Tracking Angle Analyzer")

# ═════════════════════════════════════════════════════════════════════════════
#  MODE A — ANALYSE
# ═════════════════════════════════════════════════════════════════════════════
if app_mode == "📊 Analyse":

    st.markdown(
        '<div class="banner">Software implementation of CBS VTA Meter &nbsp;·&nbsp; '
        'US Patent 4,359,768 (Abbagnaro &amp; Gust, 1982)</div>',
        unsafe_allow_html=True,
    )

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

    if run_btn:
        if not demo_mode and uploaded is None:
            st.error("Please upload a WAV file, or enable Demo mode in the sidebar.")
            st.stop()

        apply_settings()

        with st.spinner("Loading audio…"):
            channels_to_run = ['L', 'R'] if channel == 'both' else [channel]

            if demo_mode:
                fs = 44100.0
                audio_map = {
                    ch: core.generate_test_signal(
                        fs=fs, vta_error_deg=demo_error,
                        channel=ch, f_low=core.F_MOD, f_high=core.F_CARRIER,
                    )
                    for ch in channels_to_run
                }
                wav_name = "(demo/synthetic)"
            else:
                data, fs = load_wav(uploaded)
                if channel == 'both' and data.shape[1] >= 2:
                    channels_to_run = ['L', 'R']
                else:
                    channels_to_run = [channel if channel != 'both' else 'L']
                audio_map = {
                    ch: data[:, (0 if ch == 'L' else min(1, data.shape[1]-1))].copy()
                    for ch in channels_to_run
                }
                wav_name = uploaded.name
                del data   # free the full stereo array — we only need per-channel slices
                gc.collect()

        results_all = []
        progress    = st.progress(0, text="Analysing…")

        for i, (ch, audio) in enumerate(audio_map.items()):
            progress.progress(i / len(audio_map), text=f"Analysing channel {ch}…")
            old_out = sys.stdout; sys.stdout = io.StringIO()
            with warnings.catch_warnings(record=True) as caught_warns:
                warnings.simplefilter("always")
                res = core.analyse(
                    audio, fs,
                    channel=ch, f_low=core.F_MOD, f_high=core.F_CARRIER,
                    R=core.GROOVE_RADIUS, V_low=core.PEAK_VEL,
                    theta_r=core.RECORDING_ANGLE,
                    swap_channels=swap_channels,
                    modulation=core.MODULATION, verbose=True,
                )
            sys.stdout = old_out
            results_all.append(res)
            for w in caught_warns:
                if issubclass(w.category, UserWarning):
                    st.warning(str(w.message))

        progress.progress(1.0, text="Done ✓")

        ang_label = "HTA" if core.MODULATION == 'lateral' else "VTA"
        st.divider()
        st.subheader(f"📊 {ang_label} Results — {wav_name}")

        if len(results_all) >= 2:
            vta_vals = [r['vta_deg'] for r in results_all]
            vta_avg  = float(np.mean(vta_vals))
            err_avg  = vta_avg - results_all[0]['theta_r_deg']
            avg_cls  = "green" if abs(err_avg) < 2 else "amber" if abs(err_avg) < 5 else "red"
            st.markdown(
                f'<div style="text-align:center;margin-bottom:1rem;">'
                f'<span class="metric-big {avg_cls}">'
                f'AVG {ang_label}: {vta_avg:.2f}°&nbsp;&nbsp;'
                f'<span style="font-size:1.4rem">(err {err_avg:+.2f}°)</span>'
                f'</span></div>',
                unsafe_allow_html=True,
            )

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
  <hr style="border-color:#2d3548;margin:0.6rem 0">
  <div class="muted">Peak FM deviation: <span class="accent">{res['F_peak_hz']:.1f} Hz</span></div>
  <div class="muted">Phase φ: <span class="accent">{res['phi_deg']:+.1f}°</span></div>
  <div class="muted">DC tracking: <span class="accent">{res['dc_tracking']:+.4f}</span></div>
  <div class="muted">DC tracing:  <span class="accent">{res['dc_tracing']:+.4f}</span></div>
  <hr style="border-color:#2d3548;margin:0.6rem 0">
  <div class="muted">2nd harmonic: <span class="accent">{h2_db:+.1f} dB</span></div>
  <div class="muted">Tracing distortion: {td}</div>
</div>""", unsafe_allow_html=True)

        theta_r = results_all[0]['theta_r_deg']
        avg_err = float(np.mean([r['vta_error_deg'] for r in results_all]))
        if abs(avg_err) < 1.0:
            adv_col, advice = "green", f"✅ VTA is within ±1° of θR={theta_r:.1f}° — no adjustment needed."
        elif avg_err > 0:
            adv_col, advice = "amber", (f"⬆️  VTA reads {avg_err:+.1f}° too steep. "
                                        f"Lower the rear of the tonearm (or raise the front).")
        else:
            adv_col, advice = "amber", (f"⬇️  VTA reads {avg_err:+.1f}° too shallow. "
                                        f"Raise the rear of the tonearm (or lower the front).")
        st.markdown(f'<div class="result-card"><span class="{adv_col}">{advice}</span></div>',
                    unsafe_allow_html=True)

        st.subheader("🎛️ Meter Dashboard")
        fig_meter = core.plot_meter_dashboard(
            results_all, script_name="vta_analyzer_core.py",
            wav_name=wav_name, description="",
        )
        st.pyplot(fig_meter, use_container_width=True)
        plt.close(fig_meter)

        if show_spectrum:
            st.subheader("📈 Spectrum")
            fig_spec = core.plot_spectrum(audio_map, fs,
                                          f_low=core.F_MOD, f_high=core.F_CARRIER)
            st.pyplot(fig_spec, use_container_width=True)
            plt.close(fig_spec)

        # Free audio data — no longer needed after spectrum plot
        del audio_map
        gc.collect()

        if show_chain:
            st.subheader("🔗 Patent Signal Chain")
            fig_chain = core.plot_signal_chain(results_all[0])
            st.pyplot(fig_chain, use_container_width=True)
            plt.close(fig_chain)

        # Final cleanup — release results and any remaining figure memory
        plt.close('all')
        gc.collect()

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


# ═════════════════════════════════════════════════════════════════════════════
#  MODE B — CALIBRATE PEAK_VEL
# ═════════════════════════════════════════════════════════════════════════════
else:
    st.markdown(
        '<div class="cal-banner">'
        '🔧 Calibration mode — calculates the correct PEAK_VEL for your recording setup &nbsp;·&nbsp; '
        'Use the result in Analyse mode for accurate VTA readings'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
**When to use this:**  
Upload the same test-record WAV you use for VTA analysis. The calibrator finds the
PEAK_VEL value that matches the actual groove velocity on your specific pressing.
Use the result to replace the PEAK_VEL value in the core script, or note it for reference.

Two methods are available — run both when possible:
- **Method A** — WAV amplitude ratio × label carrier velocity *(CBS STR-112 only)*
- **Method B** — Back-calculation from a cartridge with a known true VTA *(any record)*
""")

    st.divider()

    # ── Calibration method picker ─────────────────────────────────────────────
    cal_method = st.radio(
        "Calibration method",
        ["Both A + B (recommended for CBS STR-112)", "Method A only", "Method B only"],
        horizontal=False,
    )
    cal_method_key = {"Both A + B (recommended for CBS STR-112)": "both",
                      "Method A only": "A",
                      "Method B only": "B"}[cal_method]

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Method A — CBS STR-112 label values**")
        ref_disp_um  = st.number_input("Reference displacement (µm)", value=11.2,
                                        min_value=0.1, max_value=100.0, step=0.1,
                                        format="%.1f",
                                        help="Carrier reference displacement from record label")
        ref_carrier_db = st.number_input("Carrier level re reference (dB)", value=-18.0,
                                          min_value=-40.0, max_value=0.0, step=0.5,
                                          format="%.1f",
                                          help="Carrier level relative to reference displacement, from label")
    with col2:
        st.markdown("**Method B — Known cartridge VTA**")
        use_known_vta = st.checkbox("I know my cartridge's true VTA", value=False)
        known_vta = None
        if use_known_vta:
            known_vta = st.number_input("Cartridge true VTA (°)", value=23.0,
                                         min_value=5.0, max_value=35.0, step=0.1,
                                         format="%.1f",
                                         help="The actual VTA of your cartridge/tonearm, independently verified")

    st.divider()

    uploaded_cal = st.file_uploader(
        "Upload the WAV recording of your test record band",
        type=["wav"],
        key="cal_uploader",
        help="Use the same recording you would use for VTA analysis.",
    )

    cal_btn = st.button("🔧  Run Calibration", type="primary", use_container_width=True)

    if cal_btn:
        if uploaded_cal is None:
            st.error("Please upload a WAV file to calibrate.")
            st.stop()

        if cal_method_key in ("B", "both") and not use_known_vta:
            st.error("Method B requires your cartridge's true VTA. "
                     "Tick 'I know my cartridge's true VTA' and enter the value.")
            st.stop()

        apply_settings()

        with st.spinner("Running calibration — this may take 20–30 seconds…"):
            # Write WAV to temp file (calibrate_peak_vel needs a file path)
            raw = uploaded_cal.read()
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(raw)
                    tmp_path = tmp.name
                del raw
                gc.collect()

                # Capture printed output from calibrate_peak_vel
                captured = io.StringIO()
                old_out  = sys.stdout
                sys.stdout = captured

                try:
                    cal_result = core.calibrate_peak_vel(
                        wav_path             = tmp_path,
                        known_vta_deg        = known_vta,
                        f_low                = core.F_MOD,
                        f_high               = core.F_CARRIER,
                        theta_r              = core.RECORDING_ANGLE,
                        ref_displacement_um  = ref_disp_um,
                        ref_carrier_db       = ref_carrier_db,
                        calibration_method   = cal_method_key,
                    )
                except Exception as e:
                    sys.stdout = old_out
                    st.error(f"Calibration failed: {e}")
                    st.stop()
                finally:
                    sys.stdout = old_out

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                gc.collect()

            log_text = captured.getvalue()

        # ── Display results ───────────────────────────────────────────────────
        st.divider()
        st.subheader("🔧 Calibration Results")

        rec = cal_result.get('recommended')
        pv_a = cal_result.get('peak_vel_A')
        pv_b = cal_result.get('peak_vel_B')
        agr  = cal_result.get('agreement_pct')

        # ── Big recommended value ─────────────────────────────────────────────
        if rec is not None:
            st.markdown(
                f'<div class="cal-card">'
                f'<div class="muted">Recommended PEAK_VEL to use</div>'
                f'<div class="metric-big amber">{rec:.4f} m/s</div>'
                f'<div class="muted">({rec*100:.3f} cm/s)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Method results side by side ───────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            if pv_a is not None:
                st.markdown(f"""
<div class="result-card">
  <div class="muted">Method A — WAV amplitude ratio</div>
  <div class="metric-med accent">{pv_a:.4f} m/s</div>
  <div class="muted">{pv_a*100:.3f} cm/s</div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-card"><div class="muted">Method A — not run</div></div>',
                            unsafe_allow_html=True)

        with c2:
            if pv_b is not None:
                st.markdown(f"""
<div class="result-card">
  <div class="muted">Method B — VTA back-calculation</div>
  <div class="metric-med accent">{pv_b:.4f} m/s</div>
  <div class="muted">{pv_b*100:.3f} cm/s</div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-card"><div class="muted">Method B — not run</div></div>',
                            unsafe_allow_html=True)

        with c3:
            if agr is not None:
                agr_cls = "green" if agr < 3 else "amber" if agr < 7 else "red"
                agr_txt = ("Excellent" if agr < 3 else "Good" if agr < 7 else "Poor — use Method B")
                st.markdown(f"""
<div class="result-card">
  <div class="muted">Agreement A vs B</div>
  <div class="metric-med {agr_cls}">{agr:.1f}%</div>
  <div class="muted">{agr_txt}</div>
</div>""", unsafe_allow_html=True)

        # ── Speed / wow summary ───────────────────────────────────────────────
        wow   = cal_result.get('wow_pp_pct', 0)
        speed = cal_result.get('speed_error_pct', 0)
        f_lo_act  = cal_result.get('f_low_actual', core.F_MOD)
        f_hi_act  = cal_result.get('f_high_actual', core.F_CARRIER)
        wow_cls   = "green" if wow < 0.1 else "amber" if wow < 0.3 else "red"
        spd_cls   = "green" if abs(speed) < 0.5 else "amber"

        st.markdown(f"""
<div class="result-card">
  <div class="muted">Turntable measurements</div>
  <div class="muted">Actual f_low: <span class="accent">{f_lo_act:.2f} Hz</span>
    &nbsp;·&nbsp; Actual f_high: <span class="accent">{f_hi_act:.2f} Hz</span></div>
  <div class="muted">Speed error: <span class="{spd_cls}">{speed:+.2f}%</span>
    &nbsp;·&nbsp; Wow (p-p): <span class="{wow_cls}">{wow:.2f}%</span></div>
</div>""", unsafe_allow_html=True)

        # ── PEAK_VEL table for all bands ──────────────────────────────────────
        if rec is not None:
            st.subheader("📋 PEAK_VEL for all bands")
            st.markdown("*Copy these values into the core script or into Custom mode.*")

            rows = []
            for db_off, label in [(0, "+6dB"), (3, "+9dB"), (6, "+12dB")]:
                v = rec * 10**(db_off / 20)
                rows.append({"Band": f"{int(core.F_MOD)}Hz {label}",
                             "PEAK_VEL (m/s)": f"{v:.4f}",
                             "PEAK_VEL (cm/s)": f"{v*100:.3f}"})
            if abs(core.F_MOD - 400) < 1:
                for db_off, label in [(0, "+6dB"), (3, "+9dB"), (6, "+12dB")]:
                    v = rec * 0.5 * 10**(db_off / 20)
                    rows.append({"Band": f"200Hz {label}",
                                 "PEAK_VEL (m/s)": f"{v:.4f}",
                                 "PEAK_VEL (cm/s)": f"{v*100:.3f}"})
            st.table(rows)

        # ── Full log (collapsible) ────────────────────────────────────────────
        with st.expander("📄 Full calibration log"):
            st.code(log_text, language="text")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div class="muted" style="text-align:center">'
    'VTA Analyzer &nbsp;·&nbsp; Based on White &amp; Gust (1979) &amp; US Patent 4,359,768 &nbsp;·&nbsp; '
    'Built with Streamlit</div>',
    unsafe_allow_html=True,
)
