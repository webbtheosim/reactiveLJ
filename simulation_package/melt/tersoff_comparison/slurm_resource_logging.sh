#!/bin/bash

RESOURCE_MONITOR_INTERVAL_SECONDS=${RESOURCE_MONITOR_INTERVAL_SECONDS:-5}
RESOURCE_START_EPOCH=
RESOURCE_TIME_LOG=
RESOURCE_GPU_LOG=
RESOURCE_SUMMARY_LOG=
RESOURCE_GPU_SAMPLER_PID=

resource_summary_line() {
    local line="$1"
    echo "${line}"
    if [[ -n "${RESOURCE_SUMMARY_LOG}" ]]; then
        echo "${line}" >> "${RESOURCE_SUMMARY_LOG}"
    fi
}

resource_logging_init() {
    local log_dir="$1"
    local model="$2"
    local epsilon="$3"
    local replicate="$4"
    local job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
    local task_id="${SLURM_ARRAY_TASK_ID:-0}"
    local prefix="${model}_resource_${job_id}_${task_id}"

    mkdir -p "${log_dir}"
    RESOURCE_TIME_LOG="${log_dir}/${prefix}_time.txt"
    RESOURCE_GPU_LOG="${log_dir}/${prefix}_gpu.csv"
    RESOURCE_SUMMARY_LOG="${log_dir}/${prefix}_summary.txt"
    : > "${RESOURCE_SUMMARY_LOG}"
    RESOURCE_START_EPOCH=$(date +%s)

    resource_summary_line "Resource_model=${model}"
    resource_summary_line "Resource_epsilon=${epsilon}"
    resource_summary_line "Resource_replicate=${replicate}"
    resource_summary_line "Resource_job_id=${job_id}"
    resource_summary_line "Resource_task_id=${task_id}"
    resource_summary_line "Resource_start_epoch=${RESOURCE_START_EPOCH}"
    resource_summary_line "Resource_start_time=$(date --iso-8601=seconds)"
    resource_summary_line "Resource_summary_log=${RESOURCE_SUMMARY_LOG}"
    resource_summary_line "Resource_time_log=${RESOURCE_TIME_LOG}"
    resource_summary_line "Resource_gpu_log=${RESOURCE_GPU_LOG}"
    resource_summary_line "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
}

resource_start_gpu_sampler() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        resource_summary_line "Gpu_memory_logging_unavailable=nvidia-smi_not_found"
        return 0
    fi

    echo "timestamp,gpu_index,memory_used_mib,memory_total_mib,utilization_gpu_percent" > "${RESOURCE_GPU_LOG}"
    (
        while true; do
            nvidia-smi \
                --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
                --format=csv,noheader,nounits >> "${RESOURCE_GPU_LOG}" 2>/dev/null || true
            sleep "${RESOURCE_MONITOR_INTERVAL_SECONDS}" || exit 0
        done
    ) &
    RESOURCE_GPU_SAMPLER_PID=$!
    resource_summary_line "Gpu_memory_sample_interval_seconds=${RESOURCE_MONITOR_INTERVAL_SECONDS}"
    resource_summary_line "Gpu_memory_sampler_pid=${RESOURCE_GPU_SAMPLER_PID}"
}

resource_stop_gpu_sampler() {
    if [[ -n "${RESOURCE_GPU_SAMPLER_PID}" ]]; then
        kill "${RESOURCE_GPU_SAMPLER_PID}" 2>/dev/null || true
        wait "${RESOURCE_GPU_SAMPLER_PID}" 2>/dev/null || true
        RESOURCE_GPU_SAMPLER_PID=
    fi
}

resource_print_time_summary() {
    if [[ ! -f "${RESOURCE_TIME_LOG}" ]]; then
        resource_summary_line "Host_memory_logging_unavailable=time_log_missing"
        return 0
    fi

    local max_rss_kib
    local time_wall
    local user_seconds
    local system_seconds
    max_rss_kib=$(awk -F: '/Maximum resident set size/ {gsub(/[ \t]/, "", $2); print $2}' "${RESOURCE_TIME_LOG}" | tail -n 1)
    time_wall=$(awk '/Elapsed \(wall clock\) time/ {sub(/^.*\):[ \t]*/, "", $0); print $0}' "${RESOURCE_TIME_LOG}" | tail -n 1)
    user_seconds=$(awk -F: '/User time \(seconds\)/ {gsub(/[ \t]/, "", $2); print $2}' "${RESOURCE_TIME_LOG}" | tail -n 1)
    system_seconds=$(awk -F: '/System time \(seconds\)/ {gsub(/[ \t]/, "", $2); print $2}' "${RESOURCE_TIME_LOG}" | tail -n 1)

    if [[ -n "${max_rss_kib}" ]]; then
        resource_summary_line "Host_memory_peak_kib=${max_rss_kib}"
        resource_summary_line "Host_memory_peak_mib=$(awk -v kib="${max_rss_kib}" 'BEGIN {printf "%.3f", kib / 1024.0}')"
    fi
    if [[ -n "${time_wall}" ]]; then
        resource_summary_line "Time_command_wall=${time_wall}"
    fi
    if [[ -n "${user_seconds}" ]]; then
        resource_summary_line "Time_user_seconds=${user_seconds}"
    fi
    if [[ -n "${system_seconds}" ]]; then
        resource_summary_line "Time_system_seconds=${system_seconds}"
    fi
}

resource_print_gpu_summary() {
    if [[ ! -f "${RESOURCE_GPU_LOG}" ]]; then
        return 0
    fi

    local sample_count
    local peak_mib
    sample_count=$(awk 'NR > 1 {count++} END {print count + 0}' "${RESOURCE_GPU_LOG}")
    peak_mib=$(awk -F, '
        NR > 1 {
            value = $3
            gsub(/^[ \t]+|[ \t]+$/, "", value)
            if ((value + 0) > max) {
                max = value + 0
            }
        }
        END {
            if (NR > 1) {
                printf "%.0f", max
            }
        }
    ' "${RESOURCE_GPU_LOG}")

    resource_summary_line "Gpu_memory_sample_count=${sample_count}"
    if [[ -n "${peak_mib}" ]]; then
        resource_summary_line "Gpu_memory_peak_mib=${peak_mib}"
    fi
}

resource_finish_logging() {
    local status="$1"
    local end_epoch
    end_epoch=$(date +%s)

    resource_stop_gpu_sampler
    resource_summary_line "Resource_end_epoch=${end_epoch}"
    resource_summary_line "Resource_end_time=$(date --iso-8601=seconds)"
    if [[ -n "${RESOURCE_START_EPOCH}" ]]; then
        resource_summary_line "Job_walltime_seconds=$((end_epoch - RESOURCE_START_EPOCH))"
    fi
    resource_summary_line "Python_exit_status=${status}"
    resource_print_time_summary
    resource_print_gpu_summary
}

resource_run_command() {
    resource_start_gpu_sampler

    local status
    set +e
    if [[ -x /usr/bin/time ]]; then
        /usr/bin/time -v -o "${RESOURCE_TIME_LOG}" "$@"
        status=$?
    else
        resource_summary_line "Host_memory_logging_unavailable=/usr/bin/time_not_found"
        "$@"
        status=$?
    fi
    set -e

    resource_finish_logging "${status}"
    return "${status}"
}

trap resource_stop_gpu_sampler EXIT
