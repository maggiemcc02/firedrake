# CHat GPT Code to quit my code if it exceed 200 GiB RAM on shared machine
#!/usr/bin/env python3

import argparse
from datetime import datetime
import os
import signal
import subprocess
import sys
import time
import psutil


def process_tree_rss(pid):
    """Return RAM usage in bytes for the process and all descendants."""
    try:
        root = psutil.Process(pid)
        processes = [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0

    total = 0

    for process in processes:
        try:
            total += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return total


def terminate_process_group(process):
    """Terminate the launched command and all MPI descendants."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit-gib",
        type=float,
        default=200.0,
        help="RAM limit in GiB",
    )

    parser.add_argument(
        "--log-every-hours",
        type=float,
        default=1.0,
        help="RAM logging interval in hours",
    )

    parser.add_argument(
        "--log-file",
        default=None,
        help="optional file for RAM usage logs",
    )

    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command to run after --",
    )

    args = parser.parse_args()

    command = args.command

    if command and command[0] == "--":
        command = command[1:]

    if not command:
        parser.error("provide a command after --")

    limit_bytes = int(args.limit_gib * 1024**3)

    # Put mpiexec and all its children in one process group.
    process = subprocess.Popen(
        command,
        start_new_session=True,
    )

    log_interval = args.log_every_hours * 3600.0
    last_log = 0.0

    def log_usage(rss_gib):
        timestamp = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

        message = (
            f"{timestamp} RAM usage: {rss_gib:.2f} GiB; "
            f"limit: {args.limit_gib:.2f} GiB"
        )

        print(message, file=sys.stderr, flush=True)

        if args.log_file is not None:
            with open(args.log_file, "a") as log:
                log.write(message + "\n")

    try:
        while process.poll() is None:
            rss_bytes = process_tree_rss(process.pid)
            rss_gib = rss_bytes / 1024**3

            now = time.monotonic()

            if now - last_log >= log_interval:
                log_usage(rss_gib)
                last_log = now

            if rss_bytes >= limit_bytes:
                print(
                    f"RAM guard: process tree reached "
                    f"{rss_gib:.2f} GiB. Terminating.",
                    file=sys.stderr,
                    flush=True,
                )

                terminate_process_group(process)
                return 137

            time.sleep(1.0)

    except KeyboardInterrupt:
        terminate_process_group(process)
        return 130

    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())