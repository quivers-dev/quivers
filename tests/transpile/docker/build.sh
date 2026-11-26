#!/usr/bin/env bash
# Build the panproto-test-* images used by the Tier-3 numeric
# equivalence suite: those named as arguments, or all of them. Idempotent: rebuilding an image with no
# changes is a fast cache hit. Run once on a fresh checkout; the
# numeric tests will skip cells whose image is absent.
set -euo pipefail

cd "$(dirname "$0")"

images=("$@")
if [ ${#images[@]} -eq 0 ]; then
    images=(stan numpyro pyro pymc edward2 julia node jags bugs)
fi

for img in "${images[@]}"; do
    echo "=== building panproto-test-${img} ==="
    docker build -t "panproto-test-${img}" "${img}/"
done

echo "=== images built ==="
docker images | grep panproto-test-
