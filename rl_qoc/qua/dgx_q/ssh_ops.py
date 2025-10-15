"""
SSH utilities for deploying and running DGX-side programs that coordinate with QUA jobs.

This module provides two layers of helpers:

1) Low-level primitives (simple, composable)
   - ensure_remote_dir(user, host, remote_dir): create a directory on the remote host.
   - upload_text(user, host, remote_path, content): write a text file remotely.
   - upload_file(user, host, local_path, remote_path): copy a local file to the remote host.
   - run_remote_python(user, host, python_bin, script_path, extra_args): run a Python script on the remote host.

2) Higher-level orchestration (batteries-included)
   - upload_directory(user, host, local_dir, remote_dir): sync a whole directory (rsync with scp fallback).
   - write_remote_file(user, host, remote_path, content, make_executable): write and optionally chmod a file.
   - run_remote_command(user, host, command, env, cwd, ssh_options): run any remote command and capture output.
   - stream_remote_command(user, host, command, ssh_options): run and stream output to a Queue for live logs.
   - deploy_and_run_script(...): one-shot ensure-dir → write-file → execute (streaming or blocking).

Relationship to generate_dgx_program.py
--------------------------------------
`rl_qoc/qua/dgx_q/generate_dgx_program.py` builds a Python script that runs on the DGX to orchestrate
training with the local QM backend. You can either:

- Generate the DGX program locally, read its contents, then push it using write_remote_file/deploy_and_run_script.
- Or construct your own remote script content and push it directly.

Typical client flow
-------------------
1. Build and upload the QUA program to OPX (handled elsewhere via rl_qoc/QM environment).
2. Generate the DGX-side program (locally) using generate_dgx_program(env, ...), which returns a local path.
3. Read that file's contents and send it to the DGX using deploy_and_run_script or write_remote_file + run.
4. Stream output (optional) and let the DGX-side program synchronize with the OPX program.

Examples
--------
Deploy a generated DGX program and run it with streaming:

    from pathlib import Path
    from rl_qoc.qua.dgx_q.generate_dgx_program import generate_dgx_program
    from rl_qoc.qua.dgx_q.ssh_ops import deploy_and_run_script

    # 1) Generate program locally
    local_prog_path = generate_dgx_program(env, ppo_config, path_to_python_wrapper="/remote/opnic")
    contents = Path(local_prog_path).read_text()

    # 2) Push to DGX and run
    proc, q, remote_path = deploy_and_run_script(
        user, host,
        remote_dir="/home/ghuser/dgx_jobs/job_001",
        filename="dgx_program.py",
        content=contents,
        interpreter="python3",
        stream=True,
    )

Minimal: write and run a quick script on DGX (blocking):

    from rl_qoc.qua.dgx_q.ssh_ops import ensure_remote_dir, write_remote_file, run_remote_command

    ensure_remote_dir(user, host, "/home/ghuser/tmp")
    write_remote_file(user, host, "/home/ghuser/tmp/hello.py", "print('hello')\n")
    rc, out, err = run_remote_command(user, host, "python3 hello.py", cwd="/home/ghuser/tmp")
    assert rc == 0

When to use which helper
------------------------
- Use upload_text / upload_file when you just need to place artifacts on the DGX.
- Use run_remote_python for a simple Python script execution (no cwd/env control needed).
- Use run_remote_command when you need full control (env, cwd, ssh options) and captured output.
- Use stream_remote_command to tail logs as the remote program runs.
- Use deploy_and_run_script for the common case of writing a DGX program and executing it immediately.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Sequence, Dict
import shlex
from threading import Thread
from queue import Queue


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



# ------------ Extended utilities for DGX program deployment and execution ------------ #

def upload_directory(user: str, host: str, local_dir: str | Path, remote_dir: str | Path) -> None:
    """Upload a local directory to a remote path.

    Tries rsync (faster, incremental). Falls back to scp -r if rsync is unavailable.
    """
    local = str(local_dir)
    remote = str(remote_dir)
    target = _build_ssh_target(user, host)

    try:
        subprocess.check_call([
            "rsync", "-az", "--delete", f"{local.rstrip('/')}/", f"{target}:{remote.rstrip('/')}/"
        ])
    except (subprocess.CalledProcessError, FileNotFoundError):
        subprocess.check_call(["ssh", target, f"mkdir -p {shlex.quote(remote)}"])  # ensure exists
        subprocess.check_call(["scp", "-r", local, f"{target}:{remote}"])


def write_remote_file(user: str, host: str, remote_path: str | Path, content: str, make_executable: bool = False, chmod_mode: str = "755") -> None:
    """Create or overwrite a remote file with provided content; optionally make it executable."""
    upload_text(user, host, str(remote_path), content)
    if make_executable:
        target = _build_ssh_target(user, host)
        subprocess.check_call(["ssh", target, f"chmod {shlex.quote(chmod_mode)} {shlex.quote(str(remote_path))}"])


def run_remote_command(
    user: str,
    host: str,
    command: str,
    *,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    ssh_options: Optional[Sequence[str]] = None,
) -> Tuple[int, str, str]:
    """Run an arbitrary command on the remote host.

    Returns (returncode, stdout, stderr).
    """
    target = _build_ssh_target(user, host)
    env_prefix = " ".join([f"{k}={shlex.quote(v)}" for k, v in (env or {}).items()])
    remote_cmd = command
    if env_prefix:
        remote_cmd = f"{env_prefix} {remote_cmd}"
    if cwd:
        remote_cmd = f"cd {shlex.quote(cwd)} && {remote_cmd}"
    ssh_base = ["ssh"] + (list(ssh_options) if ssh_options else []) + [target, remote_cmd]
    proc = subprocess.Popen(
        ssh_base,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate()
    return proc.returncode, out, err


def _stream_reader(stream, queue: Queue) -> None:
    for line in iter(stream.readline, ""):
        queue.put(("msg", line))
    stream.close()


def stream_remote_command(
    user: str,
    host: str,
    command: str,
    *,
    ssh_options: Optional[Sequence[str]] = None,
) -> Tuple[subprocess.Popen, Queue]:
    """Execute a remote command and stream stdout/stderr lines into a Queue.

    Returns (process, queue). The queue receives tuples of (kind, payload) with kind='msg' or 'exit'.
    """
    target = _build_ssh_target(user, host)
    ssh_base = ["ssh"] + (list(ssh_options) if ssh_options else []) + [target, command]
    proc = subprocess.Popen(
        ssh_base,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    q: Queue = Queue()
    Thread(target=_stream_reader, args=(proc.stdout, q), daemon=True).start()
    Thread(target=_stream_reader, args=(proc.stderr, q), daemon=True).start()

    def _monitor_exit(p: subprocess.Popen, queue: Queue) -> None:
        rc = p.wait()
        queue.put(("exit", rc))

    Thread(target=_monitor_exit, args=(proc, q), daemon=True).start()
    return proc, q


def wait_for_queue_signal(queue: Queue, *, ready_substring: Optional[str] = None) -> None:
    """Block until an 'exit' arrives, or 'ready_substring' appears in a message when provided.

    If ready_substring is provided, returns when a message line contains it (case-sensitive).
    Otherwise, waits for process exit and raises if non-zero.
    """
    while True:
        kind, payload = queue.get()
        if kind == "msg":
            if ready_substring and ready_substring in payload:
                return
        elif kind == "exit":
            rc = int(payload)
            if rc != 0:
                raise RuntimeError(f"Remote process exited with code {rc}")
            return


def deploy_and_run_script(
    user: str,
    host: str,
    *,
    remote_dir: str | Path,
    filename: str,
    content: str,
    interpreter: Optional[str] = None,
    args: Optional[Sequence[str]] = None,
    make_executable: bool = False,
    stream: bool = True,
    ssh_options: Optional[Sequence[str]] = None,
) -> Tuple[Optional[subprocess.Popen], Optional[Queue], str]:
    """High-level helper: ensure dir, write script, and execute it.

    Returns (process, queue, remote_path). If stream=False, runs to completion and returns (None, None, remote_path).
    """
    ensure_remote_dir(user, host, str(remote_dir))
    remote_path = str(Path(remote_dir) / filename)
    write_remote_file(user, host, remote_path, content, make_executable=make_executable)

    cmd = remote_path if make_executable and interpreter is None else (
        f"{interpreter} {shlex.quote(remote_path)}" if interpreter else remote_path
    )
    if args:
        cmd = " ".join([cmd] + [shlex.quote(a) for a in args])

    if stream:
        proc, q = stream_remote_command(user, host, cmd, ssh_options=ssh_options)
        return proc, q, remote_path
    else:
        rc, out, err = run_remote_command(user, host, cmd, ssh_options=ssh_options)
        if rc != 0:
            raise RuntimeError(f"Remote command failed (rc={rc}):\nSTDOUT:\n{out}\nSTDERR:\n{err}")
        return None, None, remote_path

