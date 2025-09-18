#!/usr/bin/env bash
set -euo pipefail

# jika ada env DVC_NO_PULL=true maka kita skip dvc pull
if [ "${DVC_NO_PULL:-}" = "true" ]; then
  echo "DVC pull skipped (DVC_NO_PULL=true)"
else
  # jika ada file .dvc/config dan remote terkonfigurasi, coba pull
  if [ -f .dvc/config ] || [ -f .dvc/config.local ]; then
    echo "Running: dvc pull (to fetch data from remote)"
    #kalau fail kasih warning dan lanjut
    if ! dvc pull --verbose; then
      echo "Warning: dvc pull failed — continuing without pulled cache"
    fi
    # kalau gada dvc config found
  else
    echo "No .dvc config found, skipping dvc pull"
  fi
fi

#bikin dir
mkdir -p /app/artifacts /app/mlruns /app/metrics

#exec cmd
exec "$@"
