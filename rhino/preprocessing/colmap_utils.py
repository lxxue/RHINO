import subprocess
from pathlib import Path
from typing import Sequence


# modified from nerfstudio
def run_command(cmd: Sequence[str | Path], verbose=False):
    """Runs a command and returns the output.

    Args:
        cmd: Command to run.
        verbose: If True, logs the output of the command.
    Returns:
        The output of the command if return_output is True, otherwise None.
    """
    argv = [str(part) for part in cmd]
    try:
        result = subprocess.run(
            argv,
            capture_output=not verbose,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or "").strip()
        message = f"Command failed ({error.returncode}): {' '.join(argv)}"
        if details:
            message += f"\n{details}"
        raise RuntimeError(message) from error
    return result.stdout


# prepare sparse reconstruction for BlenderNeuralangelo
def undistort_images(image_dir, sparse_dir, dense_dir):
    undistortion_cmd = [
        "colmap",
        "image_undistorter",
        "--image_path",
        str(image_dir),
        "--input_path",
        str(sparse_dir),
        "--output_path",
        str(dense_dir),
    ]
    run_command(undistortion_cmd)
