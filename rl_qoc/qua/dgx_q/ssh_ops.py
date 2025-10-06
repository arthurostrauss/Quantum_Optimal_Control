from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple


def _build_ssh_target(user: str, host: str) -> str:
    return f"{user}@{host}"


def ensure_remote_dir(user: str, host: str, remote_dir: str) -> None:
    """Create a remote directory if it doesn't already exist."""
    target = _build_ssh_target(user, host)
    cmd = ["ssh", target, f"mkdir -p {remote_dir}"]
    subprocess.check_call(cmd)


def upload_text(user: str, host: str, remote_path: str, content: str) -> None:
    """Upload text content to a remote file using a temporary local file and scp."""
    remote_path = str(remote_path)
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        target = _build_ssh_target(user, host)
        subprocess.check_call(["scp", tmp_path, f"{target}:{remote_path}"])
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def upload_file(user: str, host: str, local_path: str | Path, remote_path: str | Path) -> None:
    """Upload a local file to a remote path via scp."""
    local = str(local_path)
    remote = str(remote_path)
    target = _build_ssh_target(user, host)
    subprocess.check_call(["scp", local, f"{target}:{remote}"])


def run_remote_python(user: str, host: str, python_bin: str, script_path: str, extra_args: Optional[str] = None) -> Tuple[int, str, str]:
    """Execute a remote Python script and return (returncode, stdout, stderr)."""
    target = _build_ssh_target(user, host)
    args = f" {extra_args}" if extra_args else ""
    remote_cmd = f"{python_bin} {script_path}{args}"
    proc = subprocess.Popen(
        ["ssh", target, remote_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate()
    return proc.returncode, out, err


