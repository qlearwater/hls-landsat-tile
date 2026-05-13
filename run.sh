#!/bin/bash

set -e
set -x

export PYTHONUNBUFFERED=1
export EARTHDATA_USERNAME=$(cat /etc/secrets/EARTHDATA_USERNAME)
export EARTHDATA_PASSWORD=$(cat /etc/secrets/EARTHDATA_PASSWORD)
echo "Secret files:"
find /etc/secrets -type f 2>/dev/null

echo "STARTING DPS JOB"

basedir=$(dirname "$(readlink -f "$0")")

python "${basedir}/hls_landsat_pipeline_dps.py" \
    --mgrs_tile $1 \
    --date $2

echo "JOB COMPLETE"