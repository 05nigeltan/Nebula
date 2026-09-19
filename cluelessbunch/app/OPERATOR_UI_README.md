# Operator-facing results

RailGuard now places a plain-language action card before the technical output for every subsystem.
The explanation layer does not change model predictions or the required submission CSV formats.
The active rail model is the promoted 75% full-sensor / 25% sensor-view SVM ensemble; see
`CORRUGATION_README.md` for its selection history.

Each card answers:

1. What did the model find?
2. Which item or recording is affected?
3. What verification step should happen next?
4. Why did the system produce the result?
5. What does the result not prove?

The advice is deterministic and is generated from model diagnostics in
`src/railguard/operator_ui.py`. It does not use generative AI. Technical plots, scores, thresholds,
and raw tables remain under **Engineering details**.

## Important governance boundary

The included action wording is explicitly marked as prototype guidance. A railway engineering or
safety owner must approve urgency levels, escalation roles, and work-order rules before operational
deployment. Model scores must not be converted directly into `Critical`, `Safe`, or
`Remove from service` states.

## Subsystem-specific changes

- Door inference exposes per-cycle model score, threshold margin, operation, current summary, and
  any direction warning for faithful explanations. The Door chart has been removed; the compact
  cycle table remains under Engineering details.
- ACV diagnostics use `Clear lead`, `Close call`, or `Insufficient data`; the legacy confidence
  column remains only for backward compatibility.
- SHM cards lead with the numerical damage index and explain it as a relative estimate of the
  fatigue effect from repeated stress changes. Training-reference comparisons provide context, not
  safety thresholds.
- Corrugation explanations distinguish the detected side from physical track location, which
  requires separate route metadata.

The UI does not assign a generic data-quality grade. Input validation still runs before inference;
invalid or incomplete files produce an error instead of a potentially misleading quality label.

Run checks with:

```powershell
uv run ruff check .
uv run pytest -q
```

The research basis, alternatives, and proposed operator-usability protocol are retained locally
at repository-root `solution-research/operator-friendly-ui/`. These Git-ignored development notes
are not included in the submission.
