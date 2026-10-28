#!/bin/bash
# Create the working folder layout that the scripts and notebooks expect, using symbolic links into this repository.
# Usage: ./env/setup_workdir.sh /absolute/path/to/workdir      then:  export CREDIT_BN_ROOT=/absolute/path/to/workdir
# Use an empty folder. Existing symlinks of the same name are replaced; real files are never overwritten (skipped with a warning).
# Data (Kaggle cs-training.csv) and every generated file live in the working folder, never in the repository.
set -e
WD="${1:?usage: setup_workdir.sh /absolute/path/to/workdir}"
case "$WD" in /*) ;; *) echo "error: the working folder must be an absolute path: $WD" >&2; exit 1;; esac
WD="${WD%/}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ADD="$WD/Submissions/4. Article/WRITEUP"
link() {  # link SOURCE DESTINATION: skip if a real (non-symlink) file or folder is already there
  if [ -e "$2" ] && [ ! -L "$2" ]; then echo "skipped (real file exists): $2" >&2; return 0; fi
  ln -sfn "$1" "$2"
}
mkdir -p "$WD/Data/GiveMeSomeCredit" "$WD/Data/processed" "$WD/shared" "$WD/full_variable/notebooks" "$WD/selection/notebooks" \
         "$ADD/analysis_additions/code" "$ADD/figures_final/code"
link "$REPO/src" "$WD/shared/src"
link "$REPO/env" "$WD/shared/env"
for f in "$REPO"/scripts/analysis/*.py;    do link "$f" "$ADD/analysis_additions/code/$(basename "$f")"; done
for f in "$REPO"/scripts/figures/*.py;     do link "$f" "$ADD/figures_final/code/$(basename "$f")"; done
for f in "$REPO"/scripts/exploration/*.py; do link "$f" "$WD/Data/$(basename "$f")"; done
link "$REPO/notebooks/Expirements_FROZEN.ipynb" "$WD/full_variable/notebooks/Expirements_FROZEN.ipynb"
for f in "$REPO"/notebooks/[0-9]*.ipynb; do link "$f" "$WD/selection/notebooks/$(basename "$f")"; done
echo "Working folder ready: $WD"
echo "Now place cs-training.csv (from Kaggle) in $WD/Data/GiveMeSomeCredit/ and export CREDIT_BN_ROOT=$WD"
