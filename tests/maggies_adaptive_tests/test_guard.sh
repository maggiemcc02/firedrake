#!/usr/bin/env bash

PID="$1"

while kill -0 "$PID" 2>/dev/null; do
    RSS_KB=$(ps -p "$PID" -o rss= | tr -d ' ')

    # Deliberately tiny threshold so the test triggers immediately.
    if (( RSS_KB > 1 )); then
        echo "Test threshold crossed; stopping PID $PID"
        kill -TERM "$PID"
        sleep 2
        kill -0 "$PID" 2>/dev/null && kill -KILL "$PID"
        exit 0
    fi

    sleep 1
done
