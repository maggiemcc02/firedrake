
#!/usr/bin/env bash



PID=3552416



MAX_RSS_KB=$((100 * 1024 * 1024))

MIN_AVAILABLE_KB=$((100 * 1024 * 1024))

MIN_DISK_KB=$((50 * 1024 * 1024))

CHECK_SECONDS=30



descendants() {

    local parent="$1"

    local child



    echo "$parent"



    for child in $(pgrep -P "$parent" 2>/dev/null); do

        descendants "$child"

    done

}



stop_run() {

    local reason="$1"

    local pids



    echo "$(date): STOPPING RUN — $reason"



    pids=$(descendants "$PID" | tr '\n' ' ')

    kill -TERM $pids 2>/dev/null || true

    sleep 10



    if kill -0 "$PID" 2>/dev/null; then

        pids=$(descendants "$PID" | tr '\n' ' ')

        kill -KILL $pids 2>/dev/null || true

    fi



    exit 0

}



echo "$(date): guarding PID $PID"



while kill -0 "$PID" 2>/dev/null; do

    PIDS=$(descendants "$PID" | paste -sd, -)



    RSS_KB=$(ps -p "$PIDS" -o rss= |

        awk '{total += $1} END {print total + 0}')



    AVAILABLE_KB=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)



    DISK_KB=$(df -Pk /scratch |

        awk 'NR == 2 {print $4}')



    RSS_GIB=$(awk -v x="$RSS_KB" 'BEGIN {printf "%.2f", x/1048576}')

    AVAILABLE_GIB=$(awk -v x="$AVAILABLE_KB" 'BEGIN {printf "%.2f", x/1048576}')

    DISK_GIB=$(awk -v x="$DISK_KB" 'BEGIN {printf "%.2f", x/1048576}')



    echo "$(date): RSS=${RSS_GIB} GiB, available=${AVAILABLE_GIB} GiB, /scratch free=${DISK_GIB} GiB"



    if (( RSS_KB > MAX_RSS_KB )); then

        stop_run "RSS exceeded 20 GiB"

    fi



    if (( AVAILABLE_KB < MIN_AVAILABLE_KB )); then

        stop_run "available system RAM fell below 100 GiB"

    fi



    if (( DISK_KB < MIN_DISK_KB )); then

        stop_run "/scratch free space fell below 50 GiB"

    fi



    sleep "$CHECK_SECONDS"

done



echo "$(date): PID $PID is no longer running; guard exiting"

