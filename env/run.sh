#!/bin/bash
# Run a .py script with the reproducibility settings used in this project.
# PYTHONHASHSEED must be set BEFORE Python starts (hill-climbing is not reproducible otherwise).
# Usage: CREDIT_BN_ROOT=/path/to/workdir [PYTHON=/path/to/python] ./run.sh <script.py> [args]
: "${CREDIT_BN_ROOT:?set CREDIT_BN_ROOT to the working folder (see README)}"
export CREDIT_BN_ROOT
export PYTHONHASHSEED=0
PY="${PYTHON:-python}"
if [ "$1" = "" ]; then
  echo "usage: run.sh <script.py> [args]"; exit 1
fi
exec "$PY" "$@"
