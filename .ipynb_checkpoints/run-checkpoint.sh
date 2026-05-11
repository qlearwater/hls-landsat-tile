#!/bin/bash

set -e

echo "STARTING DPS JOB"

python hls_landsat_pipeline_dps.py \
    --mgrs_tile $1 \
    --date $2

echo "JOB COMPLETE"