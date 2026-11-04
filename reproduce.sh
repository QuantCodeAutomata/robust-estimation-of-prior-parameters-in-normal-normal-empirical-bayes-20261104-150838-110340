#!/usr/bin/env bash
# Reproduce every experiment of this repository.
set -uo pipefail

python -m pytest -m "not market_data"
python -m exp.exp1
python -m exp.exp2
