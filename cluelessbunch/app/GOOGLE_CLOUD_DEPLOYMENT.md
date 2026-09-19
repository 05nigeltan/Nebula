# RailGuard on Google Cloud Run

## Status and prerequisites

Live service: https://railguard-rilk77egmq-uc.a.run.app/
The homepage and health endpoint were verified publicly (HTTP 200). Full cloud inference
still requires representative uploads. The local development machine has no Docker or gcloud;
future builds and deployments use Cloud Shell/Cloud Build.
The canonical local deployment directory is now `cluelessbunch/app/`; the remote extracted
deployment ZIP still uses the `railguard-cloud-run/` directory described below.

Target project: `qwiklabs-gcp-00-2fd05f65fedb`; region: `us-central1`; service: `railguard`.
The supplied zone is not used by Cloud Run. Confirm the lab end time and public-access
policy with the organiser before relying on the URL for judging. Never store passwords or
download service-account key files. Sign in through the normal Google account UI.

## Deploy from Cloud Shell

1. Sign into Google Cloud Console using the assigned lab account and select the project.
2. Open Cloud Shell using the terminal icon at the top of the console.
3. Use Cloud Shell's Upload File menu to upload `railguard-cloud-run.zip` (not predictions.zip).
4. Extract into a fresh directory and run:

```bash
unzip railguard-cloud-run.zip
cd railguard-cloud-run
bash scripts/deploy_cloud_run.sh
```

If that extracted directory already exists, preserve it and extract the new archive elsewhere.
The script verifies the active lab account, enables required APIs, creates dedicated build
and runtime identities, assigns build-only roles, and deploys publicly. The runtime identity
has no project-level data access. If IAM or public-access policy blocks a step, stop and ask
the lab administrator; do not disable organisation restrictions. Recent IAM changes may take
a few minutes to propagate before a retry. The script prints the real HTTPS `run.app` URL
and checks `/_stcore/health`. It does not buy a domain, create another project or change billing.

Source deployment uses the Dockerfile and Cloud Build. Python 3.13 with locked dependencies
matches the local model environment. The image includes all four active artifacts, including
the promoted rail ensemble, and installs the local `railguard` package. No datasets, local
credentials, Git history, reports or virtual environments are included. The service listens
on `0.0.0.0:$PORT` and runs as an unprivileged user.

Initial settings: 2 vCPU, 4 GiB RAM, 0-1 instances, request concurrency 8, 60-minute timeout
and session affinity. These are demo defaults, not load-tested capacity guarantees. Persistent
Streamlit connections consume active resources; close unused browser sessions. Scaling to zero
introduces cold starts. Maximum instances is not a strict spending cap. Check lab quota first.

## Uploads and optional large ACV files

Direct uploads are capped at 25 MB (below the Cloud Run HTTP/1 request limit). Expanded ZIP
and batch content is capped at 128 MiB, with an individual expanded-file cap of 64 MiB.
Temporary uploads are removed after analysis; no permanent user-upload history is stored.
Only one inference job runs per process at a time; a second request receives a retry message.
This does not bound memory used by all open browser sessions, so concurrent-session testing
is still required before treating the initial resource settings as validated capacity.

Large ACV workbooks can be loaded through an optional Cloud demo library. This bypasses the
browser upload limit by downloading from Cloud Storage using the runtime service identity.
Only objects under `acv/` with `.xlsx` extensions and size at most 64 MiB are accepted.
Generation-conditional downloads prevent a changed object from bypassing the checked size.
The library is disabled until `RAILGUARD_INPUT_BUCKET` is set.

IMPORTANT: all visitors to the public app can select and analyse files in this library. Use
only organiser-approved demonstration data, never confidential/private operational recordings.
The bucket itself remains private. Provision this only if larger files are needed:

```bash
PROJECT_ID=qwiklabs-gcp-00-2fd05f65fedb
BUCKET="${PROJECT_ID}-railguard-demo"
gcloud storage buckets create "gs://${BUCKET}" --project="$PROJECT_ID" --location=us-central1 --uniform-bucket-level-access --public-access-prevention
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --member="serviceAccount:railguard-runtime@${PROJECT_ID}.iam.gserviceaccount.com" --role=roles/storage.objectViewer
# Upload an approved workbook using the Cloud Storage console into the acv/ prefix.
gcloud run services update railguard --project="$PROJECT_ID" --region=us-central1 --update-env-vars="RAILGUARD_INPUT_BUCKET=${BUCKET}"
```

Do not make a bucket public or upload the entire training dataset. Cloud object names and raw
exception details are not logged by this feature. The UI displays a generic storage error if
permissions or connectivity fail. Direct upload works without Cloud Storage configured.

## Verification

- Open the HTTPS URL in incognito and on another device without your Google login.
- Test Door, SHM, Rail Corrugation and ACV using approved sample recordings; compare outputs
  to local predictions. Check the two changed rail cases: Test13 = Normal, Test22 = Side I.
- Confirm downloads, invalid-file messages, 25 MB upload limit and optional cloud workbook path.
- Test two browser sessions and long-running SHM processing; inspect memory and request errors.
- Refresh/reconnect and confirm the app recovers. In-memory sessions can be lost on restart.
- Do not treat a health check alone as proof of successful model inference.

Logs and status:

```bash
gcloud run services describe railguard --region=us-central1 --project=qwiklabs-gcp-00-2fd05f65fedb
gcloud run services logs read railguard --region=us-central1 --project=qwiklabs-gcp-00-2fd05f65fedb --limit=50
```

If deployment fails, share the error text (not passwords/tokens). A memory failure requires
smaller batches or revised memory settings, not removal of upload protections. If a build
cannot resolve a locked package, inspect its error before changing model-library versions.

## Rollback and shutdown

Retain previous Cloud Run revisions. To roll back, identify a known-good revision in the console
and route traffic back to it. For an intentional shutdown after judging, delete only the
`railguard` service through Cloud Run; build images and any demo bucket may still incur costs.
Do not delete the lab project. Follow organiser instructions on retention and lab expiry.

## References

- https://docs.cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-streamlit-service
- https://docs.cloud.google.com/run/docs/triggering/websockets
- https://docs.cloud.google.com/run/quotas
- https://docs.cloud.google.com/run/docs/container-contract
