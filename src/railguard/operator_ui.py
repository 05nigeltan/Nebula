"""Deterministic, plain-language interpretation of RailGuard model outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

CardTone = Literal["review", "clear", "advisory", "insufficient"]
ACTION_POLICY_VERSION = "prototype-2026-09-19"


@dataclass(frozen=True)
class OperatorCard:
    """One auditable operator-facing interpretation of a model result."""

    tone: CardTone
    status: str
    headline: str
    affected_item: str
    action: str
    evidence: str
    reasons: tuple[str, ...]
    limitation: str
    source_id: str
    policy_version: str = ACTION_POLICY_VERSION

    def __post_init__(self) -> None:
        if not self.reasons:
            raise ValueError("An operator card needs at least one evidence reason")

    def as_dict(self) -> dict[str, Any]:
        return {
            "tone": self.tone,
            "status": self.status,
            "headline": self.headline,
            "affected_item": self.affected_item,
            "action": self.action,
            "evidence": self.evidence,
            "reasons": list(self.reasons),
            "limitation": self.limitation,
            "source_id": self.source_id,
            "policy_version": self.policy_version,
        }


def _percentage(value: float) -> str:
    return f"{100.0 * float(value):.0f}%"


def build_door_card(
    source_id: str,
    predictions: pd.DataFrame,
    cycle_diagnostics: pd.DataFrame,
) -> OperatorCard:
    abnormal = predictions["prediction"].eq("Abnormal resistance")
    abnormal_count = int(abnormal.sum())
    cycle_count = len(predictions)
    if abnormal_count:
        flagged = cycle_diagnostics.loc[abnormal].sort_values("score_margin", ascending=False)
        strongest = flagged.iloc[0]
        operation = str(strongest["operation"]).lower()
        movement_word = "movement" if abnormal_count == 1 else "movements"
        return OperatorCard(
            tone="review",
            status="REVIEW RECOMMENDED",
            headline=(
                f"Unusual resistance was detected in {abnormal_count} of "
                f"{cycle_count} door movements."
            ),
            affected_item=(
                f"Strongest flagged {operation} movement: "
                f"{strongest['start_time']} to {strongest['end_time']}"
            ),
            action=(
                "Inspect the highlighted movements for obstruction, friction, alignment, and "
                "motor condition using the approved door-maintenance procedure."
            ),
            evidence=(
                f"The electrical pattern in {abnormal_count} door {movement_word} looked "
                "different enough from normal movements for the system to flag it for review."
            ),
            reasons=(
                (
                    f"The electrical pattern in {abnormal_count} {movement_word} differed from "
                    "the normal door movements previously shown to the system."
                ),
                (
                    f"The strongest flagged movement was a {operation} cycle with mean motor "
                    f"current {float(strongest['current_mean']):.0f} mA."
                ),
                "The exact flagged time ranges are available in Engineering details.",
            ),
            limitation=(
                "This detects a known resistance pattern. It does not identify the mechanical "
                "cause or exclude other door faults."
            ),
            source_id=source_id,
        )
    return OperatorCard(
        tone="clear",
        status="NO TARGET ISSUE DETECTED",
        headline=f"No abnormal-resistance pattern was detected in {cycle_count} door movements.",
        affected_item="All detected door movements",
        action="Continue routine monitoring and follow scheduled door inspections.",
        evidence="All movements were similar to the normal examples used to train the system",
        reasons=(
            f"All {cycle_count} complete movements looked similar to normal door movements.",
            "Opening and closing movements were assessed separately by the model.",
        ),
        limitation=(
            "No issue detected does not prove the door is fault-free; this model only checks for "
            "the trained abnormal-resistance pattern."
        ),
        source_id=source_id,
    )


def build_acv_card(source_id: str, diagnostics: pd.DataFrame) -> OperatorCard:
    local = diagnostics.sort_values("predicted_rank")
    top = local.iloc[0]
    runner_up = local.iloc[1] if len(local) > 1 else None
    supported = bool(top["supported"])
    clear = str(top.get("evidence_separation", top.get("confidence", "low"))).casefold() in {
        "clear lead",
        "higher",
    }
    second_text = f"Car {runner_up['car']}" if runner_up is not None else "no runner-up"
    if not supported:
        tone: CardTone = "insufficient"
        status = "INSUFFICIENT DATA"
        action = "Collect more valid cooling-operation data before prioritising a car."
        evidence = "There were not enough comparable cooling readings to separate the cars."
    else:
        tone = "review"
        status = "REVIEW RECOMMENDED"
        evidence = (
            f"Clear lead — Car {top['car']} stands out from the other cars."
            if clear
            else (
                f"Close result — Car {top['car']} and {second_text} showed similar signs "
                "of poor cooling."
            )
        )
        action = (
            f"Prioritise Car {top['car']} for ACV diagnostic checks. "
            + (
                f"Because the result is close, inspect {second_text} as well."
                if not clear
                else f"If no issue is confirmed, inspect {second_text} next."
            )
        )
    return OperatorCard(
        tone=tone,
        status=status,
        headline=f"Car {top['car']} is the highest-ranked refrigerant-leak candidate.",
        affected_item=f"Primary candidate: Car {top['car']}; next candidate: {second_text}",
        action=action,
        evidence=evidence,
        reasons=(
            (
                f"Car {top['car']} was the hottest peer during "
                f"{_percentage(top['hottest_fraction'])} of comparable cooling readings."
            ),
            (
                f"It was more than 1 °C above its requested temperature during "
                f"{_percentage(top['above_setpoint_fraction'])} of those readings."
            ),
            (
                f"{second_text} was the next closest candidate, but its combined pattern of "
                "cooling problems was weaker."
            ),
        ),
        limitation=(
            "This model assumes the workbook contains one affected car and ranks cars relative "
            "to each other. It is not proof of a leak, a leak-rate estimate, or a probability."
        ),
        source_id=source_id,
    )


def build_corrugation_card(
    source_id: str,
    prediction: str,
    diagnostics: Mapping[str, Any],
) -> OperatorCard:
    if prediction == "Normal":
        return OperatorCard(
            tone="clear",
            status="NO TARGET ISSUE DETECTED",
            headline="No known rail-corrugation pattern was detected in this recording.",
            affected_item=f"Recording: {source_id}",
            action="Continue routine track monitoring and retain the recording for trend review.",
            evidence="The vibration pattern was similar to normal track recordings",
            reasons=(
                "Neither side showed a strong enough corrugation-like pattern to be flagged.",
                "The system compared vibration patterns from both sides of the recording.",
            ),
            limitation=(
                "No issue detected applies only to the patterns represented in training and does "
                "not certify track condition."
            ),
            source_id=source_id,
        )
    return OperatorCard(
        tone="review",
        status="REVIEW RECOMMENDED",
        headline=f"The vibration pattern is most consistent with corrugation on {prediction}.",
        affected_item=f"{prediction} in recording {source_id}",
        action=(
            "Use the recording name and the train's route record to find where this measurement "
            f"was taken. Then ask the track team to inspect {prediction} at that location."
        ),
        evidence=(
            f"The vibration pattern was strong enough for the system to flag {prediction} "
            "for review."
        ),
        reasons=(
            f"{prediction} showed a corrugation-like vibration pattern.",
            f"The pattern on {prediction} was stronger than on the opposite side.",
        ),
        limitation=(
            "The recording identifies a side but not the physical track location. Route details "
            "such as train number, date and time, GPS position, or track section are needed to "
            "locate it. The result must still be confirmed by inspection."
        ),
        source_id=source_id,
    )


def build_shm_card(
    source_id: str,
    prediction: float,
    diagnostics: Mapping[str, Any],
    reference: Mapping[str, float] | None,
) -> OperatorCard:
    if reference is None:
        band = "No comparison with earlier training examples is available"
        comparison = "There is no training reference available for comparing this result."
    elif prediction > float(reference["q90"]):
        band = "Among the highest levels compared with the training examples"
        comparison = "The result is higher than most recordings previously shown to the system."
    elif prediction > float(reference["q75"]):
        band = "Higher than most training examples"
        comparison = "The result is higher than most recordings previously shown to the system."
    else:
        band = "Similar to most training examples"
        comparison = "The overall result is similar to most recordings used to train the system."
    return OperatorCard(
        tone="advisory",
        status="COMPARE WITH EARLIER RECORDINGS",
        headline=f"Estimated fatigue-damage index: {prediction:.6g}",
        affected_item=f"Stress recording: {source_id}",
        action=(
            "If earlier recordings from the same train or component are available, compare them "
            "with this result. Ask an engineer to review the component if the estimated loading "
            "continues to increase."
        ),
        evidence=band,
        reasons=(
            (
                "This index estimates the fatigue effect of the repeated stress changes in this "
                "recording. A higher value means the recorded loading is expected to contribute "
                "more fatigue damage."
            ),
            comparison,
            (
                "A small number of the largest stress changes produced nearly all of the "
                "estimated fatigue effect."
            ),
            "One recording cannot show whether the component is deteriorating over time.",
        ),
        limitation=(
            "The index is not a percentage damaged, remaining-life estimate, structural-safety "
            "status, or proof of failure. Earlier recordings and engineering limits are needed "
            "to decide whether action is required."
        ),
        source_id=source_id,
    )
