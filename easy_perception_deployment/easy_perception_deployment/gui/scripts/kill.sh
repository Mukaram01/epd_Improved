#!/usr/bin/env bash
set -u

container_name="epd_test_container"
local_launch_signature="ros2 launch easy_perception_deployment run.launch.py"
showimage_signature="ros2 run image_tools showimage --ros-args --remap /image:=/easy_perception_deployment/output"
stopped_any=0
cleanup_failed=0

log() {
  echo "[kill] $*"
}

warn() {
  echo "[kill] $*" >&2
}

resolve_docker_cmd() {
  DOCKER_CMD=()
  if [[ -n "${EPD_DOCKER_CMD:-}" ]]; then
    # shellcheck disable=SC2206
    DOCKER_CMD=(${EPD_DOCKER_CMD})
  elif command -v docker >/dev/null 2>&1; then
    DOCKER_CMD=(docker)
  fi
}

stop_docker_container() {
  resolve_docker_cmd
  if [[ ${#DOCKER_CMD[@]} -eq 0 ]]; then
    log "Docker not installed; skipping container cleanup."
    return
  fi

  if ! "${DOCKER_CMD[@]}" --version >/dev/null 2>&1; then
    log "Docker command is unavailable; skipping container cleanup."
    return
  fi

  if "${DOCKER_CMD[@]}" ps --filter "name=^/${container_name}$" --format '{{.Names}}' 2>/dev/null | grep -Fxq "${container_name}"; then
    if "${DOCKER_CMD[@]}" stop "${container_name}" -t 1 >/dev/null 2>&1; then
      log "stopped: Docker container '${container_name}'."
      stopped_any=1
    else
      warn "partial cleanup: failed to stop Docker container '${container_name}'."
      cleanup_failed=1
    fi
  else
    log "already stopped: Docker container '${container_name}' is not running."
  fi
}

stop_local_launch() {
  local local_pids remaining
  local_pids="$(pgrep -f "${local_launch_signature}" || true)"
  if [[ -z "${local_pids}" ]]; then
    log "already stopped: no local EPD ros2 launch process is running."
    return
  fi

  if kill -TERM ${local_pids} 2>/dev/null; then
    log "stopping: local EPD ros2 launch process(es)."
    stopped_any=1
  else
    warn "partial cleanup: failed to signal local EPD launch process(es)."
    cleanup_failed=1
    return
  fi

  for _ in $(seq 1 20); do
    remaining="$(pgrep -f "${local_launch_signature}" || true)"
    [[ -z "${remaining}" ]] && return
    sleep 0.1
  done

  remaining="$(pgrep -f "${local_launch_signature}" || true)"
  if [[ -n "${remaining}" ]]; then
    warn "local EPD launch did not exit after SIGTERM; sending SIGKILL."
    if ! kill -KILL ${remaining} 2>/dev/null; then
      warn "partial cleanup: failed to kill remaining local EPD launch process(es)."
      cleanup_failed=1
    fi
  fi
}

stop_image_viewer() {
  local showimage_pids
  showimage_pids="$(pgrep -f "${showimage_signature}" || true)"
  if [[ -n "${showimage_pids}" ]]; then
    if pkill -f "${showimage_signature}"; then
      log "stopped: showimage process(es) for this app."
      stopped_any=1
    else
      warn "partial cleanup: failed to stop showimage process(es)."
      cleanup_failed=1
    fi
  else
    log "already stopped: no matching showimage process for this app."
  fi
}

log "Starting cleanup for easy_perception_deployment."
stop_docker_container
stop_local_launch
stop_image_viewer

if [[ "${cleanup_failed}" -eq 1 ]]; then
  warn "STATUS: PARTIAL_CLEANUP"
  exit 2
fi

if [[ "${stopped_any}" -eq 1 ]]; then
  log "STATUS: STOPPED"
else
  log "STATUS: ALREADY_STOPPED"
fi

exit 0
