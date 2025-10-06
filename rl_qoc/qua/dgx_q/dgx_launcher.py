from __future__ import annotations

"""
DGX Quantum launcher.

Workflow:
1) User code prepares a QUA program and executes it locally on OPX using the standard QM backend.
2) We generate a DGX program using generate_dgx_program (sync-like, no cloud job, sets QMConfig.path_to_python_wrapper).
3) We upload that script to the DGX host over SSH and run it with the specified remote Python binary.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .ssh_ops import ensure_remote_dir, upload_file, run_remote_python
from .generate_dgx_program import generate_dgx_program


def launch_on_dgx(
    env: Any,
    ppo_config: Dict[str, Any],
    path_to_python_wrapper: str,
    ssh_user: str,
    ssh_host: str,
    remote_dir: str,
    remote_python: str = "python3",
    remote_filename: str = "dgx_program.py",
    extra_argv: Optional[str] = None,
) -> tuple[int, str, str]:
    """Generate the DGX program with QMConfig.path_to_python_wrapper, upload, and execute it.

    Args:
        env: Environment to infer configuration from
        ppo_config: PPO configuration dictionary
        path_to_python_wrapper: Path to OPNIC wrapper on DGX
        ssh_user: SSH username for DGX access
        ssh_host: SSH hostname for DGX access
        remote_dir: Remote directory to upload the script to
        remote_python: Python binary to use on DGX (default: "python3")
        remote_filename: Name of the uploaded script file (default: "dgx_program.py")
        extra_argv: Optional extra command line arguments to pass to the remote script

    Returns:
        (returncode, stdout, stderr) from the remote process
    """
    # Generate a local DGX program file and upload it
    local_dir = tempfile.mkdtemp(prefix="dgx_prog_")
    local_path = generate_dgx_program(
        env=env,
        ppo_config=ppo_config,
        path_to_python_wrapper=path_to_python_wrapper,
        output_dir=local_dir,
    )

    ensure_remote_dir(ssh_user, ssh_host, remote_dir)
    remote_path = str(Path(remote_dir) / remote_filename)
    upload_file(ssh_user, ssh_host, local_path, remote_path)
    return run_remote_python(ssh_user, ssh_host, remote_python, remote_path, extra_args=extra_argv)


