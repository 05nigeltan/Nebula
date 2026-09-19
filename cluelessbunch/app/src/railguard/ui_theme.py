"""Railway operations visual system for the Streamlit interface."""

from __future__ import annotations

import streamlit as st

SUBSYSTEMS = {
    "Door": {
        "nav": "01  Doors",
        "code": "DR",
        "title": "Door resistance monitor",
        "description": "Check each opening and closing movement for unusual resistance.",
        "formats": "CSV / Excel",
    },
    "Structural Health Monitoring": {
        "nav": "02  Structural health",
        "code": "SHM",
        "title": "Structural fatigue monitor",
        "description": "Estimate the fatigue effect of repeated stress loading.",
        "formats": "CSV / ZIP",
    },
    "Rail Corrugation": {
        "nav": "03  Rail corrugation",
        "code": "RC",
        "title": "Rail corrugation localizer",
        "description": "Identify corrugation-like vibration patterns and the affected side.",
        "formats": "CSV / ZIP",
    },
    "ACV Refrigerant Leak": {
        "nav": "04  ACV cooling",
        "code": "ACV",
        "title": "ACV refrigerant-leak localizer",
        "description": "Rank train cars by signs of reduced cooling performance.",
        "formats": "Excel",
    },
}


def inject_transport_theme() -> None:
    """Apply the self-contained RailGuard visual theme."""

    st.markdown(
        """
        <style>
        :root {
            --rg-ink: #102a3b;
            --rg-navy: #071d2e;
            --rg-navy-2: #0c3047;
            --rg-teal: #007f86;
            --rg-teal-soft: #dff3f1;
            --rg-amber: #ffb703;
            --rg-paper: #f4f7f8;
            --rg-line: #d7e1e5;
            --rg-muted: #5c6f7a;
        }

        html, body, [class*="css"] {
            font-family: Inter, "Segoe UI", Arial, sans-serif;
            color: var(--rg-ink);
        }

        [data-testid="stAppViewContainer"] {
            background:
                linear-gradient(90deg, rgba(7, 29, 46, 0.025) 1px, transparent 1px) 0 0 / 42px 42px,
                linear-gradient(rgba(7, 29, 46, 0.025) 1px, transparent 1px) 0 0 / 42px 42px,
                var(--rg-paper);
        }

        [data-testid="stHeader"] {
            background: rgba(244, 247, 248, 0.88);
            backdrop-filter: blur(10px);
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1180px;
            padding-top: 2.1rem;
            padding-bottom: 4rem;
        }

        [data-testid="stSidebar"] {
            background: var(--rg-navy);
            border-right: 1px solid rgba(255, 255, 255, 0.09);
        }

        [data-testid="stSidebar"] * {
            color: #edf5f7;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #b8c9d1;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 0.45rem;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label {
            min-height: 2.9rem;
            padding: 0.65rem 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 0.45rem;
            transition: background 120ms ease, border-color 120ms ease;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, 0.07);
            border-color: rgba(255, 183, 3, 0.45);
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: rgba(0, 127, 134, 0.28);
            border-color: var(--rg-amber);
            box-shadow: inset 4px 0 0 var(--rg-amber);
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] > label {
            color: #7fcbd0;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .rg-side-brand {
            border-bottom: 1px solid rgba(255, 255, 255, 0.12);
            margin: -0.3rem 0 1.25rem;
            padding: 0.25rem 0 1.2rem;
        }

        .rg-side-wordmark {
            color: #ffffff;
            font-size: 1.35rem;
            font-weight: 850;
            letter-spacing: -0.03em;
        }

        .rg-side-wordmark span { color: var(--rg-amber); }

        .rg-side-kicker {
            color: #80c9ce;
            font-size: 0.67rem;
            font-weight: 750;
            letter-spacing: 0.14em;
            margin-top: 0.22rem;
            text-transform: uppercase;
        }

        .rg-side-status {
            align-items: center;
            background: rgba(255, 255, 255, 0.055);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 0.45rem;
            display: flex;
            font-size: 0.76rem;
            gap: 0.55rem;
            margin-top: 1.4rem;
            padding: 0.72rem;
        }

        .rg-status-dot {
            background: #45d483;
            border-radius: 50%;
            box-shadow: 0 0 0 4px rgba(69, 212, 131, 0.13);
            height: 0.55rem;
            width: 0.55rem;
        }

        .rg-hero {
            background:
                radial-gradient(circle at 87% 16%, rgba(30, 180, 184, 0.24), transparent 29%),
                linear-gradient(128deg, var(--rg-navy), var(--rg-navy-2));
            border-radius: 0.65rem;
            box-shadow: 0 18px 44px rgba(7, 29, 46, 0.14);
            color: #ffffff;
            margin-bottom: 1.35rem;
            overflow: hidden;
            padding: 2.2rem 2.4rem 1.65rem;
            position: relative;
        }

        .rg-hero::before {
            background: var(--rg-amber);
            content: "";
            height: 5px;
            left: 0;
            position: absolute;
            right: 0;
            top: 0;
        }

        .rg-eyebrow {
            color: #82d2d4;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.15em;
            margin-bottom: 0.45rem;
            text-transform: uppercase;
        }

        .rg-hero h1 {
            color: #ffffff;
            font-size: clamp(2rem, 5vw, 3.5rem);
            letter-spacing: -0.055em;
            line-height: 1;
            margin: 0;
            padding: 0;
        }

        .rg-hero h1 span { color: var(--rg-amber); }

        .rg-hero-copy {
            color: #c8d8df;
            font-size: 0.96rem;
            line-height: 1.55;
            margin: 0.8rem 0 1.55rem;
            max-width: 700px;
        }

        .rg-route {
            align-items: center;
            display: grid;
            grid-template-columns: auto 1fr auto 1fr auto 1fr auto;
            max-width: 690px;
        }

        .rg-route-line {
            background: rgba(255, 255, 255, 0.3);
            height: 2px;
        }

        .rg-station {
            align-items: center;
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .rg-station-dot {
            background: var(--rg-navy);
            border: 3px solid #8bb1bc;
            border-radius: 50%;
            height: 13px;
            width: 13px;
        }

        .rg-station.active .rg-station-dot {
            background: var(--rg-amber);
            border-color: #ffffff;
            box-shadow: 0 0 0 4px rgba(255, 183, 3, 0.22);
        }

        .rg-station-label {
            color: #a8bec7;
            font-size: 0.62rem;
            font-weight: 750;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .rg-station.active .rg-station-label { color: #ffffff; }

        .rg-module-header {
            align-items: center;
            background: rgba(255, 255, 255, 0.93);
            border: 1px solid var(--rg-line);
            border-left: 5px solid var(--rg-teal);
            border-radius: 0.55rem;
            display: flex;
            gap: 1rem;
            margin: 0.4rem 0 1.1rem;
            padding: 1rem 1.15rem;
        }

        .rg-module-code {
            align-items: center;
            background: var(--rg-teal-soft);
            border-radius: 0.35rem;
            color: var(--rg-teal);
            display: flex;
            flex: 0 0 3.25rem;
            font-size: 0.82rem;
            font-weight: 900;
            height: 3.25rem;
            justify-content: center;
            letter-spacing: 0.05em;
        }

        .rg-module-copy { flex: 1; }
        .rg-module-copy h2 {
            color: var(--rg-ink);
            font-size: 1.25rem;
            letter-spacing: -0.025em;
            margin: 0;
            padding: 0;
        }

        .rg-module-copy p {
            color: var(--rg-muted);
            font-size: 0.88rem;
            margin: 0.25rem 0 0;
        }

        .rg-format-chip {
            background: #eef3f5;
            border: 1px solid var(--rg-line);
            border-radius: 999px;
            color: #3f5967;
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            padding: 0.42rem 0.7rem;
            text-transform: uppercase;
            white-space: nowrap;
        }

        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.82);
            border-radius: 0.55rem;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: #f8fbfb;
            border: 1.5px dashed #79adb1;
            border-radius: 0.55rem;
            padding: 1.2rem;
        }

        [data-testid="stFileUploaderDropzone"]:hover {
            background: #eff9f8;
            border-color: var(--rg-teal);
        }

        .stButton > button, .stDownloadButton > button {
            background: var(--rg-amber);
            border: 1px solid #e8a600;
            border-radius: 0.35rem;
            color: #142a36;
            font-weight: 800;
            min-height: 2.7rem;
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            background: #ffc533;
            border-color: #cc9200;
            color: #071d2e;
        }

        [data-testid="stAlert"] {
            border-radius: 0.45rem;
            border-width: 0 0 0 5px;
            box-shadow: 0 7px 18px rgba(16, 42, 59, 0.06);
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--rg-line);
            border-radius: 0.45rem;
            padding: 0.8rem 0.9rem;
        }

        [data-testid="stMetricLabel"] {
            color: var(--rg-muted);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        [data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid var(--rg-line);
            border-radius: 0.5rem;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--rg-line);
            border-radius: 0.4rem;
            overflow: hidden;
        }

        .rg-disclaimer {
            border-top: 1px solid var(--rg-line);
            color: #6b7c85;
            font-size: 0.72rem;
            line-height: 1.5;
            margin-top: 2rem;
            padding-top: 1rem;
        }

        @media (max-width: 700px) {
            [data-testid="stMainBlockContainer"] { padding-top: 1rem; }
            .rg-hero { padding: 1.75rem 1.2rem 1.3rem; }
            .rg-hero-copy { font-size: 0.88rem; }
            .rg-station-label { display: none; }
            .rg-module-header { align-items: flex-start; flex-wrap: wrap; }
            .rg-format-chip { margin-left: 4.25rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand(available_models: int) -> None:
    """Render the compact control-centre identity in the sidebar."""

    st.sidebar.markdown(
        f"""
        <div class="rg-side-brand">
            <div class="rg-side-wordmark"><span>Rail</span>Guard</div>
            <div class="rg-side-kicker">Fleet condition control</div>
        </div>
        <div class="rg-side-status">
            <span class="rg-status-dot"></span>
            <span>{available_models} of 4 monitoring models ready</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(active_subsystem: str) -> None:
    """Render a transport-network-inspired product header."""

    station_keys = tuple(SUBSYSTEMS)
    station_labels = ("Doors", "Structure", "Rail", "ACV")
    stations = []
    for key, label in zip(station_keys, station_labels, strict=True):
        active = " active" if key == active_subsystem else ""
        stations.append(
            f'<div class="rg-station{active}"><span class="rg-station-dot"></span>'
            f'<span class="rg-station-label">{label}</span></div>'
        )
    route = '<span class="rg-route-line"></span>'.join(stations)
    st.markdown(
        f"""
        <section class="rg-hero">
            <div class="rg-eyebrow">Rail operations intelligence</div>
            <h1><span>Rail</span>Guard</h1>
            <p class="rg-hero-copy">
                One control point for condition signals across doors, structures, track and
                onboard cooling systems.
            </p>
            <div class="rg-route" aria-label="Four monitored train systems">{route}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_module_header(subsystem: str) -> None:
    """Render the selected monitoring module and its accepted file format."""

    metadata = SUBSYSTEMS[subsystem]
    st.markdown(
        f"""
        <section class="rg-module-header">
            <div class="rg-module-code">{metadata["code"]}</div>
            <div class="rg-module-copy">
                <h2>{metadata["title"]}</h2>
                <p>{metadata["description"]}</p>
            </div>
            <div class="rg-format-chip">Accepts {metadata["formats"]}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render the prototype-use boundary consistently on every module."""

    st.markdown(
        """
        <div class="rg-disclaimer">
            <strong>Prototype decision support.</strong> RailGuard highlights patterns for
            maintenance review. It does not replace inspection, engineering judgement, or
            approved railway operating procedures.
        </div>
        """,
        unsafe_allow_html=True,
    )
