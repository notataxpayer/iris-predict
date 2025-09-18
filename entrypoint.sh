#!/usr/bin/env bash
set -euo pipefail

# jika ada env DVC_NO_PULL=true maka kita skip dvc pull
if [ "${DVC_NO_PULL:-}" = "true" ]; then
  echo "DVC pull skipped (DVC_NO_PULL=true)"
else
  # jika ada file .dvc/config dan remote terkonfigurasi, coba pull
  if [ -f .dvc/config ] || [ -f .dvc/config.local ]; then
    echo "Running: dvc pull (to fetch data from remote)"
    # run dvc pull but don't fail the whole container if there's no remote reachable
    # we try once; if it fails we print a warning but continue to run the CMD anyway
    if ! dvc pull --verbose; then
      echo "Warning: dvc pull failed — continuing without pulled cache"
    fi
  else
    echo "No .dvc config found, skipping dvc pull"
  fi
fi

# ensure artifacts + mlruns dirs exist and are writable
mkdir -p /app/artifacts /app/mlruns /app/metrics

# then exec the container CMD (so signals are forwarded)
exec "$@"
