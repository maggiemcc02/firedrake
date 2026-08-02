#!/usr/bin/env bash

# Limits in KiB.
MAX_RSS_KB=$((120 * 1024 * 1024))       # 120 GiB for this job
MIN_AVAILABLE_KB=$((100 * 1024 * 1024)) # Keep 100 GiB free system-wide
MIN_DISK_KB=$((50 * 1024 * 1024))       # Keep 50 GiB free on /scratch

CHECK_EVERY=300  # five minutes
PID_FILE="parallel_outerloop_output/run.pid"
WATCH_DIR="/scratch"

stop_job() {
    echo "$(date): Stopping parallel_outerloop_maxwell.py"

    if [[ -f "$PID_FILE" ]]; then
        MPIPID=$(cat "$PID_FILE")
        kill -TERM "$MPIPID" 2>/dev/null || true
    fi

    pkill -TERM -u "$USER" -f '[p]arallel_outerloop_maxwell\.py' \
        2>/dev/null || true

    sleep 20

    if [[ -f "$PID_FILE" ]]; then
        kill -KILL "$(cat "$PID_FILE")" 2>/dev/null || true
    fi

    pkill -KILL -u "$USER" -f '[p]arallel_outerloop_maxwell\.py' \
        2>/dev/null || true
}

while true; do
    PIDS=$(pgrep -u "$USER" -f '[p]arallel_outerloop_maxwell\.py' || true)

    if [[ -z "$PIDS" ]]; then
        echo "$(date): No matching Python ranks remain. Watchdog exiting."
        exit 0
    fi

    PID_LIST=$(echo "$PIDS" | paste -sd, -)

    RSS_KB=$(
        ps -o rss= -p "$PID_LIST" |
        awk '{total += $1} END {print total + 0}'
    )

    AVAILABLE_KB=$(
        awk '/MemAvailable:/ {print $2}' /proc/meminfo
    )

    DISK_KB=$(
        df -Pk "$WATCH_DIR" |
        awk 'NR == 2 {print $4}'
    )

    printf '%s: ranks=%s, RSS=%.2f GiB, available=%.2f GiB, disk=%.2f GiB\n' \
        "$(date)" \
        "$(echo "$PIDS" | wc -w)" \
        "$(awk "BEGIN {print $RSS_KB/1024/1024}")" \
        "$(awk "BEGIN {print $AVAILABLE_KB/1024/1024}")" \
        "$(awk "BEGIN {print $DISK_KB/1024/1024}")"

    if (( RSS_KB > MAX_RSS_KB )); then
        echo "$(date): Job exceeded its RAM limit."
        stop_job
        exit 1
    fi

    if (( AVAILABLE_KB < MIN_AVAILABLE_KB )); then
        echo "$(date): Machine available memory is too low."
        stop_job
        exit 1
    fi

    if (( DISK_KB < MIN_DISK_KB )); then
        echo "$(date): /scratch disk space is too low."
        stop_job
        exit 1
    fi

    sleep "$CHECK_EVERY"
done
