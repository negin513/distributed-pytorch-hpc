import os
import sys


def redirect(*, log_prefix="std", rank_env="RANK", path="."):
    """
    Redirect stdout and stderr to files based on the process rank.
    This captures Python output and low-level native output.

    Args:
        log_prefix (str): Prefix for the output file names.
        rank_env (str): Environment variable name that holds the rank ID.
    """
    rank = os.environ.get(rank_env, "0")
    stdout_path = f"{path}/{log_prefix}.{rank}.out"
    stderr_path = f"{path}/{log_prefix}.{rank}.err"

    stdout_file = open(stdout_path, "w")
    stderr_file = open(stderr_path, "w")

    # Redirect Python-level streams
    sys.stdout = stdout_file
    sys.stderr = stderr_file

    # Redirect underlying OS-level file descriptors
    os.dup2(stdout_file.fileno(), 1)  # FD 1 is stdout
    os.dup2(stderr_file.fileno(), 2)  # FD 2 is stderr

    # Optional: Flush Python buffers aggressively
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        # For Python < 3.7
        raise AssertionError("Python version must be >= 3.7")

    # Optional: return file handles if you want to flush or close them later
    return stdout_file, stderr_file
