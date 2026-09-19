"""Bound expensive inference within one Streamlit server process."""

from threading import BoundedSemaphore

from railguard.uploads import UploadDataError

_INFERENCE_SLOT = BoundedSemaphore(1)


def run_analysis(function, *args, **kwargs):
    if not _INFERENCE_SLOT.acquire(blocking=False):
        raise UploadDataError("Another analysis is running. Wait a moment, then retry your upload.")
    try:
        return function(*args, **kwargs)
    finally:
        _INFERENCE_SLOT.release()
