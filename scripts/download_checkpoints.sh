#!/usr/bin/env bash
# Download GINIO checkpoint archives from the GitHub Release and verify their SHA-256.
#
#   bash scripts/download_checkpoints.sh                 # every archive listed in SHA256SUMS
#   bash scripts/download_checkpoints.sh nanobench_ginio_rd_t119 tlio_ginio_w_ln189
#   GINIO_RELEASE_TAG=v1.0.0 bash scripts/download_checkpoints.sh
#
# Archives are extracted into checkpoints/<name>/ (folder names match checkpoints/MANIFEST.md).
set -euo pipefail
REPO="${GINIO_REPO:-chankyo123/GINIO}"
TAG="${GINIO_RELEASE_TAG:-v1.0.0}"
BASE="https://github.com/${REPO}/releases/download/${TAG}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/checkpoints"
mkdir -p "${DEST}"
cd "${DEST}"

echo "Fetching SHA256SUMS from ${BASE}"
curl -fsSL -o SHA256SUMS "${BASE}/SHA256SUMS"

if [ "$#" -gt 0 ]; then
  wanted=("$@")
else
  mapfile -t wanted < <(awk '{print $2}' SHA256SUMS | sed 's/\.zip$//; s/\.tar\.gz$//')
fi

for name in "${wanted[@]}"; do
  asset="$(awk -v n="${name}" '$2==n".zip" || $2==n".tar.gz" {print $2}' SHA256SUMS | head -1)"
  if [ -z "${asset}" ]; then echo "!! ${name}: not in SHA256SUMS of ${TAG}"; continue; fi
  echo "-> ${asset}"
  curl -fL --retry 3 -o "${asset}" "${BASE}/${asset}"
  grep " ${asset}\$" SHA256SUMS | sha256sum -c -
  case "${asset}" in
    *.zip)    unzip -qo "${asset}" ;;
    *.tar.gz) tar -xzf "${asset}" ;;
  esac
  rm -f "${asset}"
done
echo "Done. Checkpoints are under ${DEST}/<name>/ ; see checkpoints/MANIFEST.md for what each one reproduces."
