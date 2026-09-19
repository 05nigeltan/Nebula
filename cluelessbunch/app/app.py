"""Streamlit demonstration UI for the four RailGuard subsystem models."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import rainflow
import streamlit as st

from railguard.acv.inference import predict_acv_files
from railguard.acv.parsing import AcvDataError
from railguard.cloud_inputs import cloud_location, download_cloud_workbook, list_cloud_workbooks
from railguard.corrugation.inference import predict_corrugation_files
from railguard.corrugation.parsing import CorrugationDataError
from railguard.door.inference import predict_door_detailed
from railguard.door.parsing import (
    EXCEL_SUFFIXES,
    DoorDataError,
)
from railguard.operator_ui import (
    OperatorCard,
    build_acv_card,
    build_corrugation_card,
    build_door_card,
    build_shm_card,
)
from railguard.runtime import run_analysis
from railguard.shm.inference import load_shm_artifact, predict_shm_files
from railguard.shm.parsing import ShmDataError, load_shm_file
from railguard.ui_theme import (
    SUBSYSTEMS,
    inject_transport_theme,
    render_footer,
    render_hero,
    render_module_header,
    render_sidebar_brand,
)
from railguard.uploads import UploadDataError, materialize_uploads

ROOT = Path(__file__).resolve().parent
DOOR_MODEL_PATH = ROOT / "artifacts" / "door" / "model.joblib"
SHM_MODEL_PATH = ROOT / "artifacts" / "shm" / "model.joblib"
CORRUGATION_MODEL_PATH = ROOT / "artifacts" / "corrugation" / "model.joblib"
ACV_MODEL_PATH = ROOT / "artifacts" / "acv" / "model.joblib"


def style_chart(figure):
    """Apply the RailGuard visual language to engineering charts."""

    figure.update_layout(
        template="plotly_white",
        colorway=["#007f86", "#ffb703", "#38627a", "#7fb7ba"],
        font={"color": "#102a3b", "family": "Inter, Segoe UI, Arial, sans-serif"},
        title={"font": {"color": "#102a3b", "size": 18}},
        margin={"l": 28, "r": 20, "t": 58, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        legend_title_text="",
    )
    figure.update_xaxes(gridcolor="#e7eef0", linecolor="#c9d7dc")
    figure.update_yaxes(gridcolor="#e7eef0", linecolor="#c9d7dc")
    return figure


def render_operator_card(card: OperatorCard) -> None:
    """Render the decision first and keep technical data separate."""

    message = f"**{card.status}**\n\n{card.headline}"
    if card.tone == "review":
        st.warning(message)
    elif card.tone == "clear":
        st.success(message)
    elif card.tone == "insufficient":
        st.error(message)
    else:
        st.info(message)

    with st.container(border=True):
        st.markdown(f"**Affected item:** {card.affected_item}")
        st.markdown("**Recommended next step**")
        st.write(card.action)
        st.markdown(f"**Evidence**\n\n{card.evidence}")
        st.markdown("**Why the system produced this result**")
        st.markdown("\n".join(f"- {reason}" for reason in card.reasons))
        st.caption(f"Important limitation: {card.limitation}")
        st.caption(
            f"Source: {card.source_id} · Guidance policy: {card.policy_version} "
            "(prototype—not an approved maintenance instruction)"
        )


def render_door_app() -> None:
    render_module_header("Door")
    if not DOOR_MODEL_PATH.exists():
        st.error("No Door model found. Run `uv run python scripts/train_door.py` first.")
        return
    uploaded = st.file_uploader(
        "Door sensor stream", type=["csv", "xlsx", "xls"], key="door_upload"
    )
    if uploaded is None:
        return
    try:
        suffix = Path(uploaded.name).suffix.lower()
        sheet_name: str | int = 0
        if suffix in EXCEL_SUFFIXES:
            with pd.ExcelFile(uploaded) as workbook:
                worksheets = workbook.sheet_names
            uploaded.seek(0)
            sheet_name = st.selectbox("Worksheet containing Door sensor data", worksheets)
        with tempfile.TemporaryDirectory(prefix="railguard-door-") as temporary:
            path = materialize_uploads(
                [uploaded],
                Path(temporary),
                allowed_suffixes={".csv", ".xlsx", ".xls"},
                subsystem="Door",
            )[0]
            predictions, diagnostics, cycle_diagnostics = run_analysis(
                predict_door_detailed,
                path,
                DOOR_MODEL_PATH,
                sheet_name=sheet_name,
            )
    except (DoorDataError, UploadDataError, ImportError, OSError, ValueError, KeyError) as exc:
        st.error(f"The file could not be processed: {exc}")
        return

    render_operator_card(build_door_card(Path(uploaded.name).name, predictions, cycle_diagnostics))
    st.download_button(
        "Download door_predictions.csv",
        predictions.to_csv(index=False).encode("utf-8"),
        file_name="door_predictions.csv",
        mime="text/csv",
    )
    with st.expander("Engineering details", expanded=False):
        st.caption(
            "The internal comparison score shows how unusual each movement looked to the "
            "software. The internal flagging level is a software cutoff—not a safety limit or "
            "a measurement of fault severity."
        )
        abnormal = int((predictions["prediction"] == "Abnormal resistance").sum())
        col1, col2, col3 = st.columns(3)
        col1.metric("Detected cycles", diagnostics.cycle_count)
        col2.metric("Abnormal cycles", abnormal)
        col3.metric("Normal cycles", diagnostics.cycle_count - abnormal)
        door_details = predictions.merge(cycle_diagnostics, on=["start_time", "end_time"]).rename(
            columns={
                "model_score": "internal_comparison_score",
                "detection_threshold": "internal_flagging_level",
                "score_margin": "difference_from_flagging_level",
            }
        )
        st.dataframe(
            door_details,
            width="stretch",
            hide_index=True,
        )


def _damage_distribution(values: np.ndarray) -> pd.DataFrame:
    cycles = list(rainflow.extract_cycles(values))
    amplitudes = np.asarray([cycle[0] / 2.0 for cycle in cycles], dtype=float)
    counts = np.asarray([cycle[2] for cycle in cycles], dtype=float)
    damage = counts * amplitudes**5
    bins = np.asarray([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, np.inf])
    indices = np.clip(np.digitize(amplitudes, bins) - 1, 0, len(bins) - 2)
    totals = np.bincount(indices, weights=damage, minlength=len(bins) - 1)
    labels = [f"{bins[index]:g}–{bins[index + 1]:g}" for index in range(len(bins) - 2)]
    labels.append("64+")
    shares = totals / np.sum(totals) if np.sum(totals) > 0 else totals
    return pd.DataFrame({"amplitude_bin": labels, "damage_share": shares})


def render_shm_app() -> None:
    render_module_header("Structural Health Monitoring")
    if not SHM_MODEL_PATH.exists():
        st.error("No SHM model found. Run `uv run python scripts/train_shm.py` first.")
        return
    uploads = st.file_uploader(
        "SHM stress files",
        type=["csv", "zip"],
        accept_multiple_files=True,
        key="shm_upload",
    )
    if not uploads:
        return
    try:
        with tempfile.TemporaryDirectory(prefix="railguard-shm-") as temporary:
            paths = materialize_uploads(
                uploads,
                Path(temporary),
                allowed_suffixes={".csv"},
                subsystem="SHM",
                allow_zip=True,
            )
            predictions, diagnostics = run_analysis(predict_shm_files, paths, SHM_MODEL_PATH)
            path_by_name = {path.name: path for path in paths}
            metadata = load_shm_artifact(SHM_MODEL_PATH)["metadata"]
            selected = st.selectbox("Choose a recording to review", predictions["file_id"].tolist())
            selected_prediction = float(
                predictions.loc[predictions["file_id"].eq(selected), "prediction"].iloc[0]
            )
            selected_diagnostics = diagnostics.loc[diagnostics["file_id"].eq(selected)].iloc[0]
            render_operator_card(
                build_shm_card(
                    selected,
                    selected_prediction,
                    selected_diagnostics,
                    metadata.get("target_reference"),
                )
            )
            st.download_button(
                "Download shm_predictions.csv",
                predictions.to_csv(index=False).encode("utf-8"),
                file_name="shm_predictions.csv",
                mime="text/csv",
            )

            engineering = st.expander("Engineering details", expanded=False)
            engineering.caption(
                "The exact damage index and stress-cycle measurements are provided for engineering "
                "review. The training comparison is context only; it is not a structural-safety "
                "limit or a remaining-life estimate."
            )
            col1, col2, col3 = engineering.columns(3)
            col1.metric("Files", len(predictions))
            col2.metric("Minimum damage", f"{predictions['prediction'].min():.6f}")
            col3.metric("Maximum damage", f"{predictions['prediction'].max():.6f}")
            engineering.dataframe(predictions, width="stretch", hide_index=True)
            signal = load_shm_file(path_by_name[selected])
            display_step = max(1, signal.sample_count // 5_000)
            trace = pd.DataFrame(
                {
                    "sample": np.arange(0, signal.sample_count, display_step),
                    "stress": signal.values[::display_step],
                }
            )
            stress_figure = px.line(
                trace,
                x="sample",
                y="stress",
                title=f"Stress trace — {selected}",
                color_discrete_sequence=["#007f86"],
            )
            engineering.plotly_chart(style_chart(stress_figure), width="stretch")
            contribution = _damage_distribution(signal.values)
            damage_figure = px.bar(
                contribution,
                x="amplitude_bin",
                y="damage_share",
                title="Fifth-power damage contribution by stress-amplitude bin",
                color_discrete_sequence=["#007f86"],
            )
            engineering.plotly_chart(style_chart(damage_figure), width="stretch")
            engineering.dataframe(
                diagnostics.loc[diagnostics["file_id"].eq(selected)],
                width="stretch",
                hide_index=True,
            )
    except (ShmDataError, UploadDataError, OSError, ValueError, KeyError) as exc:
        st.error(f"The SHM files could not be processed: {exc}")
        return


def render_corrugation_app() -> None:
    render_module_header("Rail Corrugation")
    if not CORRUGATION_MODEL_PATH.exists():
        st.error(
            "No Corrugation model found. Run `uv run python scripts/train_corrugation.py` first."
        )
        return
    uploads = st.file_uploader(
        "Rail Corrugation recordings",
        type=["csv", "zip"],
        accept_multiple_files=True,
        key="corrugation_upload",
    )
    if not uploads:
        return
    try:
        with tempfile.TemporaryDirectory(prefix="railguard-corrugation-") as temporary:
            paths = materialize_uploads(
                uploads,
                Path(temporary),
                allowed_suffixes={".csv"},
                subsystem="Corrugation",
                allow_zip=True,
            )
            predictions, diagnostics = run_analysis(
                predict_corrugation_files, paths, CORRUGATION_MODEL_PATH
            )
            selected = st.selectbox(
                "Choose a recording to review",
                predictions["file_id"].tolist(),
                key="rail_file",
            )
            selected_prediction = str(
                predictions.loc[predictions["file_id"].eq(selected), "prediction"].iloc[0]
            )
            selected_row = diagnostics.loc[diagnostics["file_id"].eq(selected)].iloc[0]
            render_operator_card(
                build_corrugation_card(selected, selected_prediction, selected_row)
            )
            st.download_button(
                "Download rail_predictions.csv",
                predictions.to_csv(index=False).encode("utf-8"),
                file_name="rail_predictions.csv",
                mime="text/csv",
            )

            engineering = st.expander("Engineering details", expanded=False)
            engineering.caption(
                "The internal scores show how strongly each side matched patterns in the training "
                "data. The amount above the internal flagging level is not a probability or a "
                "measurement of corrugation severity."
            )
            normal = int((predictions["prediction"] == "Normal").sum())
            side_i = int((predictions["prediction"] == "Side I").sum())
            side_ii = int((predictions["prediction"] == "Side II").sum())
            col1, col2, col3, col4 = engineering.columns(4)
            col1.metric("Files", len(predictions))
            col2.metric("Normal", normal)
            col3.metric("Side I", side_i)
            col4.metric("Side II", side_ii)
            engineering.dataframe(predictions, width="stretch", hide_index=True)
            score_frame = pd.DataFrame(
                {
                    "side": ["Side I", "Side II"],
                    "score": [
                        selected_row["side_i_adjusted_score"],
                        selected_row["side_ii_score"],
                    ],
                }
            )
            score_figure = px.bar(
                score_frame,
                x="side",
                y="score",
                title=f"Side scores — {selected}",
                color="side",
                color_discrete_map={"Side I": "#007f86", "Side II": "#ffb703"},
            )
            engineering.plotly_chart(style_chart(score_figure), width="stretch")
            engineering.caption(
                f"Decision threshold: {selected_row['decision_threshold']:.3f}; "
                f"estimated speed: {selected_row['speed_mps']:.2f} m/s."
            )
            engineering.dataframe(diagnostics, width="stretch", hide_index=True)
    except (CorrugationDataError, UploadDataError, OSError, ValueError, KeyError) as exc:
        st.error(f"The Corrugation files could not be processed: {exc}")


def render_acv_app() -> None:
    render_module_header("ACV Refrigerant Leak")
    if not ACV_MODEL_PATH.exists():
        st.error("No ACV model found. Run `uv run python scripts/train_acv.py` first.")
        return
    cloud_bucket, _ = cloud_location()
    mode = (
        st.radio("Workbook source", ["Upload from device", "Cloud demo library"], horizontal=True)
        if cloud_bucket
        else "Upload from device"
    )
    if mode == "Cloud demo library":
        st.caption(
            "Approved demonstration files only. Files in this library are available to all app visitors."
        )
        try:
            options = list_cloud_workbooks()
            if not options:
                st.info("No approved ACV workbooks are available in the cloud library yet.")
                return
            selected_cloud = st.selectbox("Cloud workbook", options)
            if not st.checkbox("Load this cloud workbook", key="acv_cloud_load"):
                return
            uploads = [download_cloud_workbook(selected_cloud)]
        except Exception:  # noqa: BLE001 -- sanitize external storage/auth errors at the UI boundary
            st.error(
                "The cloud workbook could not be loaded. Ask the administrator to check storage access."
            )
            return
    else:
        st.caption(
            "For files above the direct-upload limit, use the administrator's Cloud demo library."
        )
        uploads = st.file_uploader(
            "ACV telemetry workbooks",
            type=["xlsx"],
            accept_multiple_files=True,
            key="acv_upload",
        )
    if not uploads:
        return
    names = [Path(upload.name).name for upload in uploads]
    if len(names) != len(set(names)):
        st.error("ACV workbook filenames must be unique.")
        return
    try:
        with tempfile.TemporaryDirectory(prefix="railguard-acv-") as temporary:
            paths = materialize_uploads(
                uploads,
                Path(temporary),
                allowed_suffixes={".xlsx"},
                subsystem="ACV",
            )
            predictions, diagnostics = run_analysis(predict_acv_files, paths, ACV_MODEL_PATH)
            selected = st.selectbox(
                "Choose a workbook to review",
                predictions["file_id"].tolist(),
                key="acv_file",
            )
            local = diagnostics.loc[diagnostics["file_id"].eq(selected)].sort_values(
                "predicted_rank"
            )
            render_operator_card(build_acv_card(selected, local))
            st.download_button(
                "Download acv_predictions.csv",
                predictions.to_csv(index=False).encode("utf-8"),
                file_name="acv_predictions.csv",
                mime="text/csv",
            )

            engineering = st.expander("Engineering details", expanded=False)
            score_gap = float(local["top_score_margin"].iloc[0])
            engineering.caption(
                f"Score gap: {score_gap:.3f}. This is the difference between the combined scores "
                "of the first- and second-ranked cars. A larger gap means the first car stands out "
                "more clearly; it is not a probability, leak size, or fault severity."
            )
            col1, col2, col3 = engineering.columns(3)
            col1.metric("Workbooks", len(predictions))
            col2.metric("Cars ranked", len(diagnostics))
            col3.metric(
                "Clear-lead files",
                diagnostics.loc[
                    diagnostics["evidence_separation"].eq("Clear lead"), "file_id"
                ].nunique(),
            )
            engineering.dataframe(predictions, width="stretch", hide_index=True)
            acv_figure = px.bar(
                local,
                x="car",
                y="fault_score",
                color="evidence_separation",
                title=f"Relative ACV fault scores — {selected}",
                color_discrete_sequence=["#007f86", "#ffb703", "#38627a"],
            )
            engineering.plotly_chart(style_chart(acv_figure), width="stretch")
            engineering.dataframe(
                local[
                    [
                        "car",
                        "predicted_rank",
                        "fault_score",
                        "usable_samples",
                        "usable_fraction",
                        "evidence_separation",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
    except (AcvDataError, UploadDataError, ImportError, OSError, ValueError, KeyError) as exc:
        st.error(f"The ACV workbooks could not be processed: {exc}")


st.set_page_config(
    page_title="RailGuard | Fleet Condition Control",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="auto",
)
inject_transport_theme()
available_models = sum(
    path.exists()
    for path in (DOOR_MODEL_PATH, SHM_MODEL_PATH, CORRUGATION_MODEL_PATH, ACV_MODEL_PATH)
)
render_sidebar_brand(available_models)
subsystem = st.sidebar.radio(
    "Monitoring line",
    tuple(SUBSYSTEMS),
    format_func=lambda value: SUBSYSTEMS[value]["nav"],
)
st.sidebar.caption("Select a system, upload its telemetry, then follow the operator action card.")
render_hero(subsystem)
if subsystem == "Door":
    render_door_app()
elif subsystem == "Structural Health Monitoring":
    render_shm_app()
elif subsystem == "Rail Corrugation":
    render_corrugation_app()
else:
    render_acv_app()
render_footer()
