import os
import sys
from datetime import datetime


class Tee:
    """Mirror stdout writes to both the original stream and a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def open_tee_log(log_path, banner=None):
    """Append-open log_path, redirect sys.stdout to write to both console + file.

    Returns (original_stdout, log_file_handle) so the caller can restore
    stdout in a finally block via close_tee_log().
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a", buffering=1)  # line-buffered
    header = (
        f"\n{'=' * 80}\n"
        f"{banner or 'Run'} started: {datetime.now().isoformat(timespec='seconds')}\n"
        f"{'=' * 80}\n"
    )
    log_file.write(header)
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)
    return original_stdout, log_file


def close_tee_log(original_stdout, log_file):
    """Restore stdout and close the log file. Safe to call twice."""
    try:
        sys.stdout = original_stdout
    finally:
        try:
            log_file.close()
        except Exception:
            pass
