from __future__ import annotations

# core/ShellExecutor.py
"""
ShellExecutor
-------------
A robust shell command execution wrapper for Python projects.
Inspired by Warp's approach to handling subprocess execution safely.

Features:
- Run shell commands synchronously or asynchronously
- Stream stdout/stderr in real time
- Capture exit codes and results
- Timeout support
- Environment variable injection
- Error handling with custom exceptions
- Logging integration
"""

import subprocess
import threading
import logging
import shlex
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple


class ShellExecutor(Exception):
    """Raised when a shell command fails."""
    def __init__(self, command: str, exit_code: int, stdout: str, stderr: str):
        super().__init__(f"Command failed: {command} (exit code {exit_code})")
        self.command = command
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class ShellExecutor:
    """Executes shell commands with streaming, capture, and error handling."""

    def __init__(self, working_dir: Optional[Union[str, Path]] = None, env: Optional[Dict[str, str]] = None):
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.env = env or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(
        self,
        command: Union[str, List[str]],
        stream: bool = False,
        timeout: Optional[int] = None,
        check: bool = True,
    ) -> Tuple[int, str, str]:
        """
        Run a command synchronously.

        Args:
            command: Command string or list of args
            stream: If True, stream stdout/stderr live
            timeout: Timeout in seconds
            check: If True, raise error on non-zero exit

        Returns:
            (exit_code, stdout, stderr)
        """
        if isinstance(command, str):
            cmd_list = shlex.split(command)
        else:
            cmd_list = command

        self.logger.debug(f"Running command: {' '.join(cmd_list)}")
        self.logger.debug(f"Working dir: {self.working_dir}")
        process = subprocess.Popen(
            cmd_list,
            cwd=self.working_dir,
            env={**self.env, **dict(**dict())},  # allows later merging if needed
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stdout_lines = []
        stderr_lines = []

        def _stream_output(pipe, buffer, label):
            for line in iter(pipe.readline, ""):
                if line:
                    buffer.append(line)
                    if stream:
                        print(f"[{label}] {line}", end="")
            pipe.close()

        threads = []
        for pipe, buffer, label in [
            (process.stdout, stdout_lines, "STDOUT"),
            (process.stderr, stderr_lines, "STDERR"),
        ]:
            t = threading.Thread(target=_stream_output, args=(pipe, buffer, label))
            t.daemon = True
            t.start()
            threads.append(t)

        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            self.logger.error(f"Command timed out: {' '.join(cmd_list)}")
            raise

        for t in threads:
            t.join()

        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)

        if check and exit_code != 0:
            self.logger.error(f"Command failed with exit code {exit_code}")
            raise ShellExecutor(" ".join(cmd_list), exit_code, stdout, stderr)

        return exit_code, stdout, stderr

    def run_async(
        self,
        command: Union[str, List[str]],
        callback: Optional[callable] = None,
        check: bool = True,
    ) -> subprocess.Popen:
        """
        Run a command asynchronously (non-blocking).

        Args:
            command: Command string or list
            callback: Optional function to call when finished
            check: Raise error on non-zero exit if True

        Returns:
            subprocess.Popen handle
        """
        if isinstance(command, str):
            cmd_list = shlex.split(command)
        else:
            cmd_list = command

        process = subprocess.Popen(
            cmd_list,
            cwd=self.working_dir,
            env={**self.env, **dict(**dict())},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        def _monitor():
            stdout, stderr = process.communicate()
            if check and process.returncode != 0:
                raise ShellExecutor(" ".join(cmd_list), process.returncode, stdout, stderr)
            if callback:
                callback(process.returncode, stdout, stderr)

        t = threading.Thread(target=_monitor, daemon=True)
        t.start()
        return process

    def exists(self, binary: str) -> bool:
        """Check if a binary is available in PATH."""
        result = subprocess.run(["which", binary], capture_output=True, text=True)
        return result.returncode == 0