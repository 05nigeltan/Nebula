"""Safety and contract tests for hosted file uploads."""

from __future__ import annotations

import io
import zipfile

import pytest

from railguard.uploads import UploadDataError, UploadLimits, materialize_uploads


class Upload(io.BytesIO):
    def __init__(self, name: str, data: bytes) -> None:
        super().__init__(data)
        self.name = name

    def getvalue(self) -> bytes:
        return super().getvalue()


def zip_upload(name: str, members: dict[str, bytes], compression=zipfile.ZIP_DEFLATED) -> Upload:
    contents = io.BytesIO()
    with zipfile.ZipFile(contents, mode="w", compression=compression) as archive:
        for member_name, data in members.items():
            archive.writestr(member_name, data)
    return Upload(name, contents.getvalue())


def test_materializes_regular_and_nested_zip_files_safely(tmp_path) -> None:
    paths = materialize_uploads(
        [Upload("first.csv", b"1\n2\n"), zip_upload("batch.zip", {"nested/second.csv": b"3\n"})],
        tmp_path,
        allowed_suffixes={".csv"},
        subsystem="SHM",
        allow_zip=True,
    )

    assert [path.name for path in paths] == ["first.csv", "second.csv"]
    assert paths[1].parent == tmp_path
    assert paths[1].read_bytes() == b"3\n"


def test_rejects_duplicate_portable_names(tmp_path) -> None:
    with pytest.raises(UploadDataError, match="Duplicate SHM filename"):
        materialize_uploads(
            [Upload("Signal.csv", b"1\n"), Upload("signal.csv", b"2\n")],
            tmp_path,
            allowed_suffixes={".csv"},
            subsystem="SHM",
        )


def test_rejects_non_csv_archive_member(tmp_path) -> None:
    with pytest.raises(UploadDataError, match="must use one of these formats"):
        materialize_uploads(
            [zip_upload("batch.zip", {"notes.txt": b"not telemetry"})],
            tmp_path,
            allowed_suffixes={".csv"},
            subsystem="Corrugation",
            allow_zip=True,
        )


def test_rejects_excessively_compressed_archive_member(tmp_path) -> None:
    limits = UploadLimits(maximum_compression_ratio=2.0)
    with pytest.raises(UploadDataError, match="compressed suspiciously heavily"):
        materialize_uploads(
            [zip_upload("batch.zip", {"signal.csv": b"0\n" * 10_000})],
            tmp_path,
            allowed_suffixes={".csv"},
            subsystem="SHM",
            allow_zip=True,
            limits=limits,
        )


def test_rejects_oversized_regular_file(tmp_path) -> None:
    limits = UploadLimits(maximum_file_bytes=3)
    with pytest.raises(UploadDataError, match="larger than the hosted file limit"):
        materialize_uploads(
            [Upload("case.xlsx", b"1234")],
            tmp_path,
            allowed_suffixes={".xlsx"},
            subsystem="ACV",
            limits=limits,
        )
