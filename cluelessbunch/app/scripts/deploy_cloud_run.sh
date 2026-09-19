#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Deployment stopped. Preserve the error above; do not change lab organisation policies to bypass it." >&2' ERR

cd "$(dirname "$0")/.."
PROJECT_ID="qwiklabs-gcp-00-2fd05f65fedb"
REGION="us-central1"
SERVICE="railguard"
EXPECTED_ACCOUNT="student-04-a5834ee5bfdf@qwiklabs.net"
BUILD_ACCOUNT="railguard-builder@${PROJECT_ID}.iam.gserviceaccount.com"
RUNTIME_ACCOUNT="railguard-runtime@${PROJECT_ID}.iam.gserviceaccount.com"

command -v gcloud >/dev/null || { echo "Run this in Google Cloud Shell."; exit 1; }
ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)')"
if [[ "$ACTIVE_ACCOUNT" != "$EXPECTED_ACCOUNT" ]]; then
  echo "Sign into Cloud Shell with the assigned lab account before continuing."
  exit 1
fi
gcloud projects describe "$PROJECT_ID" --format='value(projectId)'
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  iam.googleapis.com --project="$PROJECT_ID"

# Isolate build privileges from the running application. No service-account key files.
for NAME in railguard-builder railguard-runtime; do
  ADDRESS="${NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
  EXISTING="$(gcloud iam service-accounts list --project="$PROJECT_ID" --filter="email=$ADDRESS" --format='value(email)')"
  if [[ -z "$EXISTING" ]]; then
    gcloud iam service-accounts create "$NAME" --project="$PROJECT_ID" --display-name="$NAME"
  fi
done
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$BUILD_ACCOUNT" --role=roles/run.builder --condition=None --quiet >/dev/null
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$BUILD_ACCOUNT" --role=roles/serviceusage.serviceUsageConsumer --condition=None --quiet >/dev/null

echo "Building and deploying; this can take several minutes and consumes lab resources."
gcloud run deploy "$SERVICE" \
  --project="$PROJECT_ID" --region="$REGION" --source=. \
  --build-service-account="projects/$PROJECT_ID/serviceAccounts/$BUILD_ACCOUNT" \
  --service-account="$RUNTIME_ACCOUNT" \
  --port=8080 --cpu=2 --memory=4Gi --concurrency=8 \
  --min-instances=0 --max-instances=1 --timeout=3600 --session-affinity \
  --allow-unauthenticated --quiet

URL="$(gcloud run services describe "$SERVICE" --project="$PROJECT_ID" --region="$REGION" --format='value(status.url)')"
echo "Cloud Run URL: $URL"
curl --fail --silent --show-error --retry 5 --retry-delay 3 "$URL/_stcore/health"
echo
echo "Open the URL in an incognito browser and test each subsystem. Confirm the lab survives through judging."
