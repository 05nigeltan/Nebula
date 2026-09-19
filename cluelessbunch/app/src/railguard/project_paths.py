"""Locate optional participant data without including it in the submission."""

import os
from pathlib import Path


def participant_root(app_root: Path) -> Path:
    """Accept an explicit PS3 directory, app-local data, or the original workspace data."""
    configured = os.environ.get("RAILGUARD_PARTICIPANT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    relative = Path("NebulaX-Hackathon-ProblemStatement/PS3")
    candidates = [app_root / relative, app_root.parent.parent / relative]
    return next((path for path in candidates if path.is_dir()), candidates[0])
