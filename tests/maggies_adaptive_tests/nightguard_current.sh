#!/usr/bin/env bash

set -u

ROOT_PID="${1:?Usage: $0 MPI_PID}"

CHECK_EVERY=60                         # Check every minute
MAX_RSS_KB=$((120 * 1024 * 1024))     # Stop above 120 GiB
MIN_AVAILABLE_KB=$((100 * 1024 * 1024)) # Keep 100 GiB available
MIN_DISK_KB=$((50 * 1024 * 1024))     # Keep 50 GiB free on /scratch

get_tree_pids() {
    local -a queue=("$1")
    local i=0
    local current child

    while (( i < ${#queue[@]} )); do
        current="${queue[$i]}"

        while read -r child; do
            [[ -n "$child" ]] && queue+=("$child")
        done < <(pgrep -P "$current" 2>/dev/null || true)

        ((i += 1))
    done

    printf '%s\n' "${queue[@]}" | sort -n -u
}

stop_job() {
    local reason="$1"
    local -a pids

    echo "$(date): $reason"
    echo "$(date): Stopping MPI job rooted at PID $ROOT_PID"

    mapfile -t pids < <(get_tree_pids "$ROOT_PID")

    # Stop children first, then mpiexec.
    printf '%s\n' "${pids[@]}" |
        sort -rn |
        xargs -r kill -TERM 2>/dev/null || true

    sleep 20

    pkill -KILL -u "$USER" -f '[p]arallel_outerloop_maxwell\.py' \
        2>/dev/null || true

    kill -KILL "$ROOT_PID" 2>/dev/null || true

    echo "$(date): Job stopped."
}

while true; do
    if ! kill -0 "$ROOT_PID" 2>/dev/null; then
        echo "$(date): MPI job is no longer running. Guard exiting."
        exit 0
    fi

    mapfile -t PIDS < <(get_tree_pids "$ROOT_PID")
    PID_LIST=$(IFS=,; echo "${PIDS[*]}")

    RSS_KB=$(
        ps -o rss= -p "$PID_LIST" 2>/dev/null |
        awk '{total += $1} END {print total + 0}'
    )

    AVAILABLE_KB=$(
        awk '/MemAvailable:/ {print $2}' /proc/meminfo
    )

    DISK_KB=$(
        df -Pk /scratch |
        awk 'NR == 2 {print $4}'
    )

    PROCESS_COUNT=${#PIDS[@]}

    RSS_GIB=$(awk -v x="$RSS_KB" 'BEGIN {printf "%.2f", x/1024/1024}')
    AVAILABLE_GIB=$(awk -v x="$AVAILABLE_KB" 'BEGIN {printf "%.2f", x/1024/1024}')
    DISK_GIB=$(awk -v x="$DISK_KB" 'BEGIN {printf "%.2f", x/1024/1024}')

    echo "$(date): processes=$PROCESS_COUNT, RSS=$RSS_GIB GiB, available=$AVAILABLE_GIB GiB, disk=$DISK_GIB GiB"

    if (( RSS_KB > MAX_RSS_KB )); then
        stop_job "Job exceeded its 120 GiB RAM limit."
        exit 1
    fi

    if (( AVAILABLE_KB < MIN_AVAILABLE_KB )); then
        stop_job "Machine available memory fell below 100 GiB."
        exit 1
    fi

    if (( DISK_KB < MIN_DISK_KB )); then
        stop_job "/scratch free space fell below 50 GiB."
        exit 1
    fi

    sleep "$CHECK_EVERY"
done
