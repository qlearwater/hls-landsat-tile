#!/bin/bash

set -e
set -x

export PYTHONUNBUFFERED=1

echo "STARTING DPS JOB"

basedir=$(dirname "$(readlink -f "$0")")

python "${basedir}/hls_landsat_pipeline_dps.py" \
    --mgrs_tile $1 \
    --date $2

echo "JOB COMPLETE"