from types import SimpleNamespace

import pytest

from railguard.cloud_inputs import (
    MAX_CLOUD_FILE_BYTES,
    approved_name,
    download_cloud_workbook,
    list_cloud_workbooks,
)
from railguard.uploads import UploadDataError


def test_cloud_names_and_disabled_state(monkeypatch):
    assert approved_name("acv/demo.xlsx")
    assert not approved_name("secret/demo.xlsx")
    assert not approved_name("acv/../secret.xlsx")
    assert not approved_name("acv/model.joblib")
    monkeypatch.delenv("RAILGUARD_INPUT_BUCKET", raising=False)
    with pytest.raises(UploadDataError):
        list_cloud_workbooks()
    with pytest.raises(UploadDataError):
        download_cloud_workbook("acv/demo.xlsx")


def test_cloud_listing_and_generation_bound_download(monkeypatch):
    monkeypatch.setenv("RAILGUARD_INPUT_BUCKET", "approved-demo-bucket")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return b"data"

    blob = SimpleNamespace(name="acv/demo.xlsx", size=4, generation=42, download_as_bytes=download)

    class Client:
        def list_blobs(self, name, **kwargs):
            assert name == "approved-demo-bucket"
            assert kwargs == {"prefix": "acv/", "max_results": 128}
            return [blob, SimpleNamespace(name="acv/large.xlsx", size=MAX_CLOUD_FILE_BYTES + 1)]

        def bucket(self, name):
            assert name == "approved-demo-bucket"
            return SimpleNamespace(get_blob=lambda name: blob)

    client = Client()
    assert list_cloud_workbooks(client) == ["acv/demo.xlsx"]
    workbook = download_cloud_workbook("acv/demo.xlsx", client)
    assert workbook.name == "demo.xlsx"
    assert workbook.getvalue() == b"data"
    assert calls == [{"if_generation_match": 42, "timeout": 60}]
    blob.size = MAX_CLOUD_FILE_BYTES + 1
    with pytest.raises(UploadDataError):
        download_cloud_workbook("acv/demo.xlsx", client)
