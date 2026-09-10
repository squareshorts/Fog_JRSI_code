#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash run_all.sh '/path/to/Filtered Data' [output_dir]" >&2
  exit 2
fi

DATA_ROOT="$1"
OUT_ROOT="${2:-work}"
mkdir -p "$OUT_ROOT"

cat analysis/reviewer_part_00.py analysis/reviewer_part_01.py analysis/reviewer_part_02.py analysis/reviewer_part_03.py > "$OUT_ROOT/reviewer_reanalysis.py"
python "$OUT_ROOT/reviewer_reanalysis.py" --data-root "$DATA_ROOT" --out "$OUT_ROOT/results"
python analysis/continuous_onset_dynamics.py --data-root "$DATA_ROOT" --out "$OUT_ROOT/continuous"
python analysis/antecedent_history_final.py --data-root "$DATA_ROOT" --out "$OUT_ROOT/antecedent"
python analysis/antecedent_history_final_v2.py --data-root "$DATA_ROOT" --out "$OUT_ROOT/antecedent"

python -m pip freeze > "$OUT_ROOT/python_environment_freeze.txt"
echo "Complete. Results written to: $OUT_ROOT"
