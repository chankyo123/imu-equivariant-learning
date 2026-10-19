#!/usr/bin/env bash
# Download GINIO checkpoint archives from the GitHub Release and verify their SHA-256.
#
#   bash scripts/download_checkpoints.sh                       # all four archives (~0.7 GB)
#   bash scripts/download_checkpoints.sh nanobench               # one benchmark: tlio | ridi | nanobench
#   bash scripts/download_checkpoints.sh results                 # paper_results.tar.gz (stored outputs behind the numbers)
#   GINIO_RELEASE_TAG=v1.0-corl2026 bash scripts/download_checkpoints.sh
#
# Checkpoint archives extract into checkpoints/<benchmark>_<row>_<variant>/ (names as in
# checkpoints/MANIFEST.md); paper_results.tar.gz extracts into checkpoints/paper_results/.
set -euo pipefail
REPO="${GINIO_REPO:-chankyo123/GINIO}"
TAG="${GINIO_RELEASE_TAG:-v1.0-corl2026}"
BASE="https://github.com/${REPO}/releases/download/${TAG}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/checkpoints"
mkdir -p "${DEST}"
cd "${DEST}"

echo "Fetching SHA256SUMS from ${BASE}"
curl -fsSL -o SHA256SUMS "${BASE}/SHA256SUMS"

if [ "$#" -gt 0 ]; then wanted=("$@"); else wanted=(tlio ridi nanobench results); fi

for name in "${wanted[@]}"; do
  case "${name}" in
    tlio|ridi|nanobench) asset="ginio_checkpoints_${name}.tar.gz"; dest="." ;;
    results)             asset="paper_results.tar.gz";             dest="paper_results"; mkdir -p "${dest}" ;;
    *) echo "!! unknown target '${name}' (use tlio | ridi | nanobench | results)"; continue ;;
  esac
  echo "-> ${asset}"
  curl -fL --retry 3 -o "${asset}" "${BASE}/${asset}"
  grep " ${asset}\$" SHA256SUMS | sha256sum -c -
  # the checkpoint archives carry a packaging-time copy of MANIFEST.md; the repo copy stays authoritative
  tar -xzf "${asset}" -C "${dest}" --exclude=MANIFEST.md
  rm -f "${asset}"
done
echo "Done. Checkpoints are under ${DEST}/<name>/ ; see checkpoints/MANIFEST.md for what each one reproduces."
