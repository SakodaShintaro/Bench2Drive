#!/bin/bash
# Evaluate a vla_streaming_rl checkpoint on a Bench2Drive route.
# Uses the vla_streaming_rl venv Python + CARLA 0.9.16.
#
# Usage:
#   run_my_eval_vla.sh <result_dir> <routes_xml>
#
#   result_dir : training result dir containing .hydra/config.yaml and checkpoint.pt
#   routes_xml : Bench2Drive route XML to evaluate on
set -eu

if [ $# -ne 2 ]; then
    echo "Usage: $0 <result_dir> <routes_xml>" >&2
    exit 1
fi

RESULT_DIR=$(readlink -f "$1")
ROUTES=$(readlink -f "$2")

# --- Config ---
VENV_PYTHON=${HOME}/work/vla_streaming_rl/.venv/bin/python
export CARLA_ROOT=${HOME}/CARLA_0.9.16
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh

REPO_ROOT=$(readlink -f "$(dirname "$0")/../..")
cd "${REPO_ROOT}"

export PYTHONPATH=${PYTHONPATH:-}
export PYTHONPATH=${CARLA_ROOT}/PythonAPI:${PYTHONPATH}
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}
export PYTHONPATH=${REPO_ROOT}/leaderboard:${PYTHONPATH}
export PYTHONPATH=${REPO_ROOT}/leaderboard/team_code:${PYTHONPATH}
export PYTHONPATH=${REPO_ROOT}/scenario_runner:${PYTHONPATH}
export SCENARIO_RUNNER_ROOT=${REPO_ROOT}/scenario_runner
export LEADERBOARD_ROOT=${REPO_ROOT}/leaderboard

TEAM_AGENT=${REPO_ROOT}/leaderboard/team_code/vla_streaming_agent.py
# Per-run output dir under RESULT_DIR (i.e. next to the checkpoint),
# stamped with start time so successive runs don't overwrite each other.
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${RESULT_DIR}/bench2drive_result_${TIMESTAMP}
mkdir -p "${OUTPUT_DIR}"

CHECKPOINT_ENDPOINT=${OUTPUT_DIR}/eval.json
SAVE_PATH=${OUTPUT_DIR}/

# Optional: record a video of the composed observation (what the policy sees)
export VLA_EVAL_VIDEO_PATH=${VLA_EVAL_VIDEO_PATH:-${OUTPUT_DIR}/run.mp4}
export VLA_EVAL_VIDEO_FPS=${VLA_EVAL_VIDEO_FPS:-20}

# Challenge settings
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=30000
export TM_PORT=50000
export DEBUG_CHALLENGE=0
export REPETITIONS=1
export RESUME=True
export IS_BENCH2DRIVE=True
export PLANNER_TYPE=only_traj
export GPU_RANK=0
export SAVE_PATH
export CARLA_RENDER_VISIBLE=1

CUDA_VISIBLE_DEVICES=${GPU_RANK} "${VENV_PYTHON}" \
  ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
  --routes=${ROUTES} \
  --repetitions=${REPETITIONS} \
  --track=${CHALLENGE_TRACK_CODENAME} \
  --checkpoint=${CHECKPOINT_ENDPOINT} \
  --agent=${TEAM_AGENT} \
  --agent-config=${RESULT_DIR} \
  --debug=${DEBUG_CHALLENGE} \
  --resume=${RESUME} \
  --port=${PORT} \
  --traffic-manager-port=${TM_PORT} \
  --gpu-rank=${GPU_RANK}
