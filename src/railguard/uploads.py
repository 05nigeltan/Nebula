"""Bounded materialisation of user uploads for the Streamlit application."""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class UploadDataError(ValueError):
    """Raised when an uploaded batch is unsafe or violates its file contract."""


@dataclass(frozen=True)
class UploadLimits:
    """Resource limits chosen for the memory-constrained hosted application."""

    maximum_files: int = 128
    maximum_file_bytes: int = 64 * 1024 * 1024
    maximum_total_bytes: int = 512 * 1024 * 1024
    maximum_compression_ratio: float = 100.0
    copy_chunk_bytes: int = 1024 * 1024


DEFAULT_UPLOAD_LIMITS = UploadLimits()


def _safe_name(raw_name: str, allowed_suffixes: frozenset[str], subsystem: str) -> str:
    name = Path(raw_name).name
    suffix = Path(name).suffix.lower()
    if not name or suffix not in allowed_suffixes:
        expected = ", ".join(sorted(allowed_suffixes))
        raise UploadDataError(f"{subsystem} inputs must use one of these formats: {expected}")
    return name


def _claim_name(name: str, names: set[str], subsystem: str) -> None:
    portable_name = name.casefold()
    if portable_name in names:
        raise UploadDataError(f"Duplicate {subsystem} filename: {name}")
    names.add(portable_name)


def _write_zip_member(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    target: Path,
    limits: UploadLimits,
) -> None:
    written = 0
    with archive.open(member) as source, target.open("xb") as destination:
        while chunk := source.read(limits.copy_chunk_bytes):
            written += len(chunk)
            if written > limits.maximum_file_bytes:
                raise UploadDataError(f"Archive member {member.filename!r} is too large")
            destination.write(chunk)
    if written != member.file_size:
        raise UploadDataError(f"Archive member {member.filename!r} has an invalid size")


def materialize_uploads(
    uploads: Iterable[Any],
    directory: Path,
    *,
    allowed_suffixes: Iterable[str],
    subsystem: str,
    allow_zip: bool = False,
    limits: UploadLimits = DEFAULT_UPLOAD_LIMITS,
) -> list[Path]:
    """Write validated uploads to a temporary directory without unsafe ZIP extraction."""

    uploaded_files = list(uploads)
    if not uploaded_files:
        raise UploadDataError(f"No {subsystem} files were supplied")
    suffixes = frozenset(suffix.lower() for suffix in allowed_suffixes)
    archive_count = sum(Path(upload.name).suffix.lower() == ".zip" for upload in uploaded_files)
    if archive_count and not allow_zip:
        raise UploadDataError(f"{subsystem} ZIP uploads are not supported")
    if archive_count > 1:
        raise UploadDataError("Upload at most one ZIP archive at a time")

    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    names: set[str] = set()
    total_bytes = 0

    for uploaded in uploaded_files:
        uploaded_name = Path(str(uploaded.name)).name
        if Path(uploaded_name).suffix.lower() != ".zip":
            name = _safe_name(uploaded_name, suffixes, subsystem)
            _claim_name(name, names, subsystem)
            data = uploaded.getvalue()
            if len(data) > limits.maximum_file_bytes:
                raise UploadDataError(f"{name} is larger than the hosted file limit")
            total_bytes += len(data)
            if total_bytes > limits.maximum_total_bytes:
                raise UploadDataError(f"The combined {subsystem} upload is too large")
            target = directory / name
            target.write_bytes(data)
            paths.append(target)
            continue

        try:
            with zipfile.ZipFile(uploaded) as archive:
                members = [member for member in archive.infolist() if not member.is_dir()]
                if not members:
                    raise UploadDataError("The uploaded ZIP is empty")
                if len(paths) + len(members) > limits.maximum_files:
                    raise UploadDataError(
                        f"A maximum of {limits.maximum_files} files can be analysed at once"
                    )
                for member in members:
                    if member.flag_bits & 0x1:
                        raise UploadDataError("Password-protected ZIP files are not supported")
                    name = _safe_name(member.filename, suffixes, subsystem)
                    _claim_name(name, names, subsystem)
                    if member.file_size > limits.maximum_file_bytes:
                        raise UploadDataError(f"Archive member {member.filename!r} is too large")
                    ratio = member.file_size / max(member.compress_size, 1)
                    if ratio > limits.maximum_compression_ratio:
                        raise UploadDataError(
                            f"Archive member {member.filename!r} is compressed suspiciously heavily"
                        )
                    total_bytes += member.file_size
                    if total_bytes > limits.maximum_total_bytes:
                        raise UploadDataError(f"The expanded {subsystem} archive is too large")
                    target = directory / name
                    _write_zip_member(archive, member, target, limits)
                    paths.append(target)
        except zipfile.BadZipFile as exc:
            raise UploadDataError("The uploaded ZIP file is invalid or damaged") from exc

    if len(paths) > limits.maximum_files:
        raise UploadDataError(f"A maximum of {limits.maximum_files} files can be analysed at once")
    return paths
