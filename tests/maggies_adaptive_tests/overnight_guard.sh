#!/usr/bin/env bash

PID=3349760
MAX_RSS_KB=$((120 * 1024 * 1024))
MIN_AVAILABLE_KB=$((100 * 1024 * 1024))
MIN_DISK_KB=$((50 * 1024 * 1024))

while kill -0 "$PID" 2>/dev/null; do
    # Stop if this PID is no longer your Maxwell program.
    ps -p "$PID" -o args= |
        grep -q "first_outerloop_maxwell.py" || exit 0

    RSS_KB=$(ps -p "$PID" -o rss= | tr -d ' ')
    AVAILABLE_KB=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    DISK_FREE_KB=$(df -Pk . | awk 'NR == 2 {print $4}')

    printf '%s RSS=%.2f GiB, available=%.2f GiB, disk_free=%.2f GiB\n' \
        "$(date)" \
        "$(awk "BEGIN {print $RSS_KB/1024/1024}")" \
        "$(awk "BEGIN {print $AVAILABLE_KB/1024/1024}")" \
        "$(awk "BEGIN {print $DISK_FREE_KB/1024/1024}")"

    if grep -q "FOR OUTER SWEEP NUMBER 3" first_outerloop.log; then
        echo "Stopping: reached z_its = 3."
        kill -TERM "$PID"
        sleep 60
        kill -0 "$PID" 2>/dev/null && kill -KILL "$PID"
        exit 0
    fi

    if (( RSS_KB > MAX_RSS_KB )); then
        echo "Stopping: process exceeded 120 GiB RAM."
        kill -TERM "$PID"
        sleep 60
        kill -0 "$PID" 2>/dev/null && kill -KILL "$PID"
        exit 1
    fi

    if (( AVAILABLE_KB < MIN_AVAILABLE_KB )); then
        echo "Stopping: machine has less than 100 GiB available RAM."
        kill -TERM "$PID"
        sleep 60
        kill -0 "$PID" 2>/dev/null && kill -KILL "$PID"
        exit 1
    fi

    if (( DISK_FREE_KB < MIN_DISK_KB )); then
        echo "Stopping: filesystem has less than 50 GiB free."
        kill -TERM "$PID"
        sleep 60
        kill -0 "$PID" 2>/dev/null && kill -KILL "$PID"
        exit 1
    fi

    sleep 300
done

echo "$(date): process finished normally."
