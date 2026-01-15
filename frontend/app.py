from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import Config
from core.simulation import PICSimulation


def apply_style() -> None:
    st.set_page_config(page_title="PICSIMU", layout="wide")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Space+Grotesk:wght@400;600&display=swap');
        :root {
            --bg-1: #f6efe4;
            --bg-2: #dbe7f2;
            --ink: #1d2327;
            --muted: #5c6b73;
            --accent: #b24a2f;
            --accent-2: #2b6f73;
            --card: rgba(255, 255, 255, 0.78);
            --border: rgba(29, 35, 39, 0.12);
            --shadow: 0 20px 60px rgba(29, 35, 39, 0.12);
        }
        .stApp {
            background: radial-gradient(circle at 12% 8%, rgba(178, 74, 47, 0.12), transparent 40%),
                        radial-gradient(circle at 86% 6%, rgba(43, 111, 115, 0.10), transparent 38%),
                        linear-gradient(135deg, var(--bg-1), var(--bg-2));
            color: var(--ink);
            font-family: "Space Grotesk", "Segoe UI", sans-serif;
        }
        h1, h2, h3, .display {
            font-family: "DM Serif Display", Georgia, serif;
            letter-spacing: 0.2px;
        }
        .hero {
            padding: 1.2rem 1.4rem;
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.7);
            border-radius: 18px;
            box-shadow: var(--shadow);
        }
        .kicker {
            text-transform: uppercase;
            font-size: 0.78rem;
            letter-spacing: 0.2rem;
            color: var(--muted);
            margin-bottom: 0.4rem;
        }
        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            box-shadow: var(--shadow);
        }
        .pill {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            background: rgba(178, 74, 47, 0.14);
            color: var(--accent);
            font-weight: 600;
            font-size: 0.8rem;
        }
        .fade-in {
            animation: fadeUp 700ms ease-out;
        }
        .stButton > button {
            background: linear-gradient(135deg, var(--accent), #d27d4d);
            color: white;
            border: none;
            border-radius: 999px;
            padding: 0.6rem 1.6rem;
            font-weight: 600;
            box-shadow: 0 10px 24px rgba(178, 74, 47, 0.3);
        }
        .stButton > button:hover {
            filter: brightness(1.05);
        }
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def plot_iv_curve(v: np.ndarray, i_total: np.ndarray, i_e: np.ndarray, i_i: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(v, i_total, color="#1d2327", linewidth=2.2, label="I_total")
    ax.plot(v, i_e, color="#b24a2f", linewidth=2.0, label="I_e (pos)")
    ax.plot(v, i_i, color="#2b6f73", linewidth=2.0, label="I_i")
    ax.set_xlabel("Probe bias (V)")
    ax.set_ylabel("Current (A)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_electron_semilog(v: np.ndarray, i_e: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.semilogy(v, np.maximum(i_e, 1.0e-30), color="#b24a2f", linewidth=2.0)
    ax.set_xlabel("Probe bias (V)")
    ax.set_ylabel("|I_e| (A)")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    return fig


def main() -> None:
    apply_style()

    st.markdown(
        """
        <div class="hero fade-in">
            <div class="kicker">1D Cylindrical PIC-MCC</div>
            <h1 class="display">PICSIMU Probe Sheath Explorer</h1>
            <p>
                High-pressure Langmuir probe simulation with cylindrical fields,
                charge exchange, and angular momentum conservation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    left, right = st.columns([1.05, 1.4], gap="large")

    with left:
        st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
        st.markdown("### Inputs")
        pressure = st.slider("Neutral pressure (Torr)", 1.0, 200.0, 50.0, step=1.0)
        density_exp = st.slider("Plasma density log10(n0) (m^-3)", 14.0, 18.0, 16.0, step=0.1)
        density = 10.0 ** density_exp
        te = st.slider("Electron temperature Te (eV)", 0.5, 10.0, 3.0, step=0.1)
        ti = st.slider("Ion temperature Ti (eV)", 0.01, 0.2, 0.03, step=0.01)
        v_start = st.slider("Start bias (V)", -100.0, 100.0, -50.0, step=1.0)
        v_end = st.slider("End bias (V)", -100.0, 100.0, 50.0, step=1.0)
        v_steps = st.slider("Voltage steps", 5, 101, 21, step=1)

        with st.expander("Advanced settings", expanded=False):
            n_particles = st.slider("Macro particles per species", 2000, 50000, 12000, step=1000)
            n_burn_in = st.slider("Burn-in steps", 0, 50000, 12000, step=1000)
            n_sampling = st.slider("Sampling steps", 500, 50000, 12000, step=1000)
            v_wall = st.slider("Wall potential (V)", -100.0, 100.0, 0.0, step=1.0)
            probe_length = st.number_input("Probe length (m)", min_value=0.0, value=0.01, step=0.001)
            semilog_e = st.checkbox("Show semilog electron current", value=False)
            st.caption("Electron current is reported as positive magnitude.")
            st.caption("Currents scale by probe length; L=1 m returns A/m.")

        run_clicked = st.button("Generate I-V Curve")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
        st.markdown("### Status")
        st.markdown('<span class="pill">Ready</span>', unsafe_allow_html=True)
        st.markdown("Run a voltage sweep to generate the I-V trace.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")

        iv_data = st.session_state.get("iv_data")
        if run_clicked:
            progress = st.progress(0.0)
            status = st.empty()

            def progress_cb(done: int, total: int, v: float) -> None:
                progress.progress(done / total)
                status.markdown(f"Scanning {v:.1f} V ({done}/{total})")

            with st.spinner("Sweeping voltage..."):
                cfg = Config(
                    N0=density,
                    Te=te,
                    Ti=ti,
                    P_Torr=pressure,
                    V_WALL=v_wall,
                )
                sim = PICSimulation(
                    cfg,
                    n_particles=n_particles,
                    v_bias=v_start,
                    probe_length=probe_length,
                )
                iv_data = sim.scan_voltage_range(
                    v_start=v_start,
                    v_end=v_end,
                    n_steps=v_steps,
                    n_burn_in=n_burn_in,
                    n_sampling=n_sampling,
                    progress_cb=progress_cb,
                )
                st.session_state["iv_data"] = iv_data

            progress.empty()
            status.markdown('<span class="pill">Sweep complete</span>', unsafe_allow_html=True)

        if iv_data is not None:
            st.markdown(
                f"<div class='pill'>Points: {len(iv_data['voltages'])}</div>",
                unsafe_allow_html=True,
            )

    if iv_data is not None:
        st.write("")
        fig_iv = plot_iv_curve(
            iv_data["voltages"],
            iv_data["I_total"],
            iv_data["I_electron"],
            iv_data["I_ion"],
        )
        st.pyplot(fig_iv, clear_figure=True)

        if semilog_e:
            fig_log = plot_electron_semilog(
                iv_data["voltages"],
                iv_data["I_electron"],
            )
            st.pyplot(fig_log, clear_figure=True)


if __name__ == "__main__":
    main()
