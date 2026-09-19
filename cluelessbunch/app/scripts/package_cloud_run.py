"""Package only deployable source, approved model files and cloud instructions."""

import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    names = [
        "Dockerfile",
        ".dockerignore",
        ".gcloudignore",
        "pyproject.toml",
        "uv.lock",
        "app.py",
        ".streamlit/config.toml",
        "GOOGLE_CLOUD_DEPLOYMENT.md",
        "scripts/deploy_cloud_run.sh",
    ]
    names += [str(p.relative_to(root)).replace("\\", "/") for p in (root / "src").rglob("*.py")]
    names += [
        f"artifacts/{model}/{name}"
        for model in ("door", "shm", "acv", "corrugation")
        for name in ("model.joblib", "metadata.json")
    ]
    output = root / "railguard-cloud-run.zip"
    if output.exists():
        raise FileExistsError(
            "Deployment archive already exists; preserve it before creating another"
        )
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(root / name, arcname=f"railguard-cloud-run/{name}")
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
    # This app directory is canonical; do not create a nested snapshot.
    print(f"Created {output} ({output.stat().st_size:,} bytes, {len(names)} files)")


if __name__ == "__main__":
    main()
