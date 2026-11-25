#!/usr/bin/env bash
# Full pipeline: download, tabulate ACS employment, refit, summarize, draw the figure (~5 min after downloads).
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/get_data.py
python -m autoscores.employment   # ACS 2012 and 2023 employment by SOC -> data/
python -m autoscores.run          # 648 refits -> results/probs.csv
python -m autoscores.analyze      # results/summary.md, results/shares.csv
python -m autoscores.figure       # results/figure1.png
