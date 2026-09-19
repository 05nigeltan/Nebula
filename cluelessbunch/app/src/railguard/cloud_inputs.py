"""Optional bounded ACV downloads from an administrator-configured GCS prefix."""

import io
import os
from pathlib import PurePosixPath

from railguard.uploads import UploadDataError

MAX_CLOUD_FILE_BYTES = 64 * 1024 * 1024


class CloudWorkbook(io.BytesIO):
    def __init__(self, name, content):
        super().__init__(content)
        self.name = name


def cloud_location():
    bucket = os.environ.get("RAILGUARD_INPUT_BUCKET", "").strip()
    # Only administrator-controlled bucket/prefix, never a user-provided URL.
    return bucket, "acv/"


def approved_name(name):
    path = PurePosixPath(name)
    return name.startswith("acv/") and path.suffix.lower() == ".xlsx" and ".." not in path.parts


def list_cloud_workbooks(client=None):
    bucket, prefix = cloud_location()
    if not bucket:
        raise UploadDataError("Cloud Storage inputs are not configured")
    if client is None:
        from google.cloud import storage

        client = storage.Client()
    result = []
    for blob in client.list_blobs(bucket, prefix=prefix, max_results=128):
        if (
            approved_name(blob.name)
            and blob.size is not None
            and 0 < blob.size <= MAX_CLOUD_FILE_BYTES
        ):
            result.append(blob.name)
    return sorted(result)


def download_cloud_workbook(name, client=None):
    bucket, _ = cloud_location()
    if not bucket or not approved_name(name):
        raise UploadDataError("Choose an approved ACV workbook")
    if client is None:
        from google.cloud import storage

        client = storage.Client()
    blob = client.bucket(bucket).get_blob(name)
    if blob is None or blob.size is None or not 0 < blob.size <= MAX_CLOUD_FILE_BYTES:
        raise UploadDataError("The workbook is missing, empty or exceeds the 64 MiB limit")
    # Freeze generation so a replacement cannot bypass the checked size.
    data = blob.download_as_bytes(if_generation_match=blob.generation, timeout=60)
    if len(data) > MAX_CLOUD_FILE_BYTES:
        raise UploadDataError("The workbook exceeds the 64 MiB limit")
    return CloudWorkbook(PurePosixPath(name).name, data)
