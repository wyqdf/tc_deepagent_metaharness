#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"

config="configs/oss120b_deepagent_opus46_5rounds_clean_official_workflow.yaml"
source_run="runs/oss120b_deepagent_opus46_5rounds_clean_official_workflow_20260524"
run_name="candidate10_retest_oss120b_keyUG_concurrency6_20260524"
max_workers="6"
clean="0"
foreground="0"

usage() {
  cat <<'USAGE'
Usage: scripts/run_candidate_retest.sh [options]

Options:
  --config PATH        Config path relative to project root.
  --source-run PATH    Source run containing val-trained memory snapshots.
  --run-name NAME      New run name under runs/.
  --max-workers N      Evaluation concurrency.
  --clean              Remove only the same retest run dir and driver log first.
  --foreground         Run in foreground instead of nohup background.
  -h, --help           Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      config="$2"
      shift 2
      ;;
    --source-run)
      source_run="$2"
      shift 2
      ;;
    --run-name)
      run_name="$2"
      shift 2
      ;;
    --max-workers)
      max_workers="$2"
      shift 2
      ;;
    --clean)
      clean="1"
      shift
      ;;
    --foreground)
      foreground="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$run_name" == *"/"* || -z "$run_name" ]]; then
  echo "Invalid --run-name: $run_name" >&2
  exit 2
fi

python_bin="$project_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python_bin="$(command -v python3)"
fi

run_dir="$project_root/runs/$run_name"
driver_log="$project_root/runs/${run_name}_driver.log"
safe_runs_root="$(realpath -m "$project_root/runs")"
safe_run_dir="$(realpath -m "$run_dir")"
safe_driver_log="$(realpath -m "$driver_log")"

case "$safe_run_dir" in
  "$safe_runs_root"/*) ;;
  *)
    echo "Refusing unsafe run dir: $safe_run_dir" >&2
    exit 2
    ;;
esac

case "$safe_driver_log" in
  "$safe_runs_root"/*) ;;
  *)
    echo "Refusing unsafe driver log: $safe_driver_log" >&2
    exit 2
    ;;
esac

cd "$project_root"
mkdir -p "$project_root/runs"

if [[ "$clean" == "1" ]]; then
  rm -rf -- "$run_dir"
  rm -f -- "$driver_log"
fi

if [[ -e "$run_dir" || -e "$driver_log" ]]; then
  echo "Refusing to reuse existing run artifacts." >&2
  echo "Run dir: $run_dir" >&2
  echo "Driver log: $driver_log" >&2
  echo "Re-run with --clean to remove only these two paths." >&2
  exit 1
fi

cmd=(
  "$python_bin"
  scripts/retest_candidates.py
  --config "$config"
  --source-run "$source_run"
  --run-name "$run_name"
  --max-workers "$max_workers"
)

echo "Project: $project_root"
echo "Run: $run_name"
echo "Max workers: $max_workers"
echo "Driver log: $driver_log"

if [[ "$foreground" == "1" ]]; then
  "${cmd[@]}" 2>&1 | tee "$driver_log"
else
  nohup "${cmd[@]}" > "$driver_log" 2>&1 &
  pid="$!"
  echo "$pid" > "$project_root/runs/${run_name}.pid"
  echo "PID: $pid"
  echo "Live log: $run_dir/live.log"
  echo "Scores: $run_dir/candidate_test_scores.json"
fi
