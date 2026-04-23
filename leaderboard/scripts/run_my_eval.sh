#!/bin/bash
# Phase 1: pipeline smoke test with NPC agent on a single dev10 route.
# Uses the vla_streaming_rl venv Python + CARLA 0.9.16.
set -eu

# --- Config ---
VENV_PYTHON=${HOME}/work/vla_streaming_rl/.venv/bin/python
export CARLA_ROOT=${HOME}/CARLA_0.9.16
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh

# From Bench2Drive root
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "${REPO_ROOT}"

export PYTHONPATH=${PYTHONPATH:-}
export PYTHONPATH=${CARLA_ROOT}/PythonAPI:${PYTHONPATH}
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}
# NOTE: 0.9.16 wheel is already installed in venv — no .egg on PYTHONPATH.
export PYTHONPATH=${REPO_ROOT}/leaderboard:${PYTHONPATH}
export PYTHONPATH=${REPO_ROOT}/leaderboard/team_code:${PYTHONPATH}
export PYTHONPATH=${REPO_ROOT}/scenario_runner:${PYTHONPATH}
export SCENARIO_RUNNER_ROOT=${REPO_ROOT}/scenario_runner
export LEADERBOARD_ROOT=${REPO_ROOT}/leaderboard

# Route / agent
ROUTES=${REPO_ROOT}/leaderboard/data/dev10_single.xml
TEAM_AGENT=${REPO_ROOT}/leaderboard/leaderboard/autoagents/npc_agent.py
TEAM_CONFIG=/dev/null   # npc_agent ignores config
CHECKPOINT_ENDPOINT=${REPO_ROOT}/eval_single.json
SAVE_PATH=${REPO_ROOT}/eval_single/
mkdir -p "${SAVE_PATH}"

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

CUDA_VISIBLE_DEVICES=${GPU_RANK} "${VENV_PYTHON}" \
  ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
  --routes=${ROUTES} \
  --repetitions=${REPETITIONS} \
  --track=${CHALLENGE_TRACK_CODENAME} \
  --checkpoint=${CHECKPOINT_ENDPOINT} \
  --agent=${TEAM_AGENT} \
  --agent-config=${TEAM_CONFIG} \
  --debug=${DEBUG_CHALLENGE} \
  --resume=${RESUME} \
  --port=${PORT} \
  --traffic-manager-port=${TM_PORT} \
  --gpu-rank=${GPU_RANK}
