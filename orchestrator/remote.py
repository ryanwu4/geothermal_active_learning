"""SSH/SCP transport for hybrid AL runs.

The local driver (``scripts/run_al_local.py``) runs in a tmux pane on bend and
pushes everything that needs to happen on Sherlock through this module. We
deliberately do **not** establish or own the ssh authentication — the user is
expected to configure ``ControlMaster auto`` + a ``ControlPath`` in
``~/.ssh/config`` for the Sherlock host, then run a one-off ``ssh sherlock`` in
a separate tmux pane to authenticate (Duo handled interactively there). Every
call here just runs ``ssh <host>`` and OpenSSH transparently multiplexes onto
the existing master socket.

We keep ``control_path`` only so we can stat() the socket up-front and produce
a friendly "open a session in your tmux pane first" error before invoking ssh.

If the socket is missing or auth has expired, calls raise :class:`SshUnavailable`
and the driver writes a ``NEEDS_AUTH.md`` notice and exits cleanly.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


class SshUnavailable(RuntimeError):
    """Raised when the ControlMaster socket is missing/dead or auth has expired.

    The driver is expected to catch this once at the top of each iteration,
    write a resume notice, and exit. Do not retry inside this module — the
    user has to re-establish the socket manually.
    """


_JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")
# ssh exits 255 for any transport-level failure (auth refused, socket gone,
# host unreachable). Other non-zero exits are command failures on the remote.
_SSH_TRANSPORT_FAILURE = 255
_AUTH_HINTS = (
    "Permission denied",
    "Connection refused",
    "Connection closed",
    "no such identity",
    "Could not resolve hostname",
    "Host key verification failed",
    "Control socket connect",
    "ControlSocket",
    "no controlling master",
    "Connection timed out",
)


@dataclass
class RemoteSession:
    """Wrapper around ssh / rsync / scp that reuses a ControlMaster socket.

    Construct once per driver run; reuse for every call. The socket itself is
    not owned by this object — the user opens and closes it externally.
    """

    host: str
    control_path: str

    def __post_init__(self) -> None:
        # Expand ~ once so we don't resurface it in every command.
        self.control_path = os.path.expanduser(self.control_path)

    # ------------------------------------------------------------------
    # Common ssh option fragments
    # ------------------------------------------------------------------
    #
    # We don't pass ``-o ControlPath=`` or ``-o ControlMaster=`` here — those
    # come from the user's ``~/.ssh/config`` for this host. ``BatchMode=yes``
    # is the only override we insist on (so a missing master fails fast
    # instead of dropping into an interactive Duo prompt).

    def _ssh_opts(self) -> list[str]:
        return ["-o", "BatchMode=yes"]

    def _ssh_argv(self) -> list[str]:
        return ["ssh", *self._ssh_opts(), self.host]

    def _rsync_ssh_arg(self) -> str:
        return "ssh -o BatchMode=yes"

    # ------------------------------------------------------------------
    # Liveness check
    # ------------------------------------------------------------------

    def check_alive(self) -> None:
        """Verify the ControlMaster socket is up. Raises SshUnavailable if not."""
        if not os.path.exists(self.control_path):
            raise SshUnavailable(
                f"ControlMaster socket not found at {self.control_path}. "
                "Open an authenticated session in a tmux pane first."
            )
        # ssh resolves ControlPath from ~/.ssh/config for this host.
        proc = subprocess.run(
            ["ssh", "-O", "check", self.host],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            raise SshUnavailable(
                f"ssh -O check failed (rc={proc.returncode}): "
                f"{(proc.stderr or proc.stdout).strip()}"
            )

    # ------------------------------------------------------------------
    # Remote command execution
    # ------------------------------------------------------------------

    def run(
        self,
        cmd: list[str] | str,
        *,
        check: bool = True,
        capture: bool = True,
        cwd: str | None = None,
    ) -> subprocess.CompletedProcess:
        """Run ``cmd`` on the remote host through the ControlMaster socket.

        ``cmd`` may be a list (will be shell-quoted) or a pre-built shell string
        (used when piping/redirection is needed). ``cwd`` optionally cd's first.
        """
        if isinstance(cmd, list):
            remote_cmd = " ".join(shlex.quote(c) for c in cmd)
        else:
            remote_cmd = cmd
        if cwd:
            remote_cmd = f"cd {shlex.quote(cwd)} && {remote_cmd}"

        argv = [*self._ssh_argv(), "--", remote_cmd]
        proc = subprocess.run(
            argv,
            capture_output=capture, text=True, check=False,
        )
        if proc.returncode == _SSH_TRANSPORT_FAILURE or _looks_like_auth_failure(proc.stderr):
            raise SshUnavailable(
                f"ssh transport failure for {self.host}: "
                f"{(proc.stderr or proc.stdout).strip()}"
            )
        if check and proc.returncode != 0:
            raise RemoteCommandError(
                argv, proc.returncode, proc.stdout, proc.stderr
            )
        return proc

    # ------------------------------------------------------------------
    # File transfer
    # ------------------------------------------------------------------

    def pull(self, remote_path: str, local_path: Path | str) -> None:
        """rsync ``host:remote_path`` → ``local_path`` (file or dir).

        ``-P`` = ``--partial --progress``: a partial file from an interrupted
        run is kept as ``.<name>.partial`` and resumed next time, but the final
        rename only happens once the transfer is complete — so consumers never
        see a half-written ``current_compiled.h5``.
        """
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            "rsync", "-aP",
            "-e", self._rsync_ssh_arg(),
            f"{self.host}:{remote_path}", str(local),
        ]
        self._run_transfer(argv)

    def push(self, local_path: Path | str, remote_path: str) -> None:
        """rsync ``local_path`` → ``host:remote_path``."""
        argv = [
            "rsync", "-aP",
            "-e", self._rsync_ssh_arg(),
            str(local_path), f"{self.host}:{remote_path}",
        ]
        self._run_transfer(argv)

    def _run_transfer(self, argv: list[str]) -> None:
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)
        # rsync exits 255 when the underlying ssh fails (transport-level), and
        # also surfaces ssh stderr in its own stderr. Treat both as auth/socket
        # loss the same way.
        if proc.returncode == _SSH_TRANSPORT_FAILURE or _looks_like_auth_failure(proc.stderr):
            raise SshUnavailable(
                f"rsync transport failure: {(proc.stderr or proc.stdout).strip()}"
            )
        if proc.returncode != 0:
            raise RemoteCommandError(argv, proc.returncode, proc.stdout, proc.stderr)

    # ------------------------------------------------------------------
    # SLURM helpers (mirror orchestrator/slurm.py over ssh)
    # ------------------------------------------------------------------

    def submit_sbatch(
        self,
        remote_sbatch_path: str,
        *,
        dependency: str | None = None,
        cwd: str | None = None,
    ) -> str:
        """sbatch on Sherlock; returns the job id as a string."""
        cmd = ["sbatch", "--parsable"]
        if dependency:
            cmd.append(f"--dependency={dependency}")
        cmd.append(remote_sbatch_path)
        proc = self.run(cmd, cwd=cwd, check=True)
        raw = proc.stdout.strip().split(";")[0]
        if not raw.isdigit():
            m = _JOB_ID_RE.search(proc.stdout)
            if not m:
                raise RemoteCommandError(
                    cmd, proc.returncode, proc.stdout, proc.stderr,
                    note="could not parse sbatch job id",
                )
            raw = m.group(1)
        return raw

    def job_state(self, job_id: str) -> str:
        """Best-effort SLURM state lookup. Returns ``UNKNOWN`` if sacct is empty.

        ``sacct`` records linger after jobs finish; ``squeue`` clears them. We
        consult sacct first (post-completion authoritative source); fall back
        to squeue for very-recently-submitted jobs that sacct hasn't indexed.
        """
        proc = self.run(
            ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2", "--allocations"],
            check=False,
        )
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line:
                # Strip trailing flags like " CANCELLED+" or " by 12345"
                return line.split()[0].rstrip("+")
        # sacct empty → try squeue
        proc = self.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"], check=False,
        )
        line = proc.stdout.strip()
        return line.split()[0] if line else "UNKNOWN"


# ----------------------------------------------------------------------
# Errors / helpers
# ----------------------------------------------------------------------


class RemoteCommandError(RuntimeError):
    """Non-zero exit from a remote command that wasn't a transport failure."""

    def __init__(self, argv, returncode, stdout, stderr, note: str | None = None):
        msg = f"remote command failed (rc={returncode}): {argv!r}"
        if note:
            msg += f" — {note}"
        if stderr:
            msg += f"\nstderr: {stderr.strip()}"
        if stdout:
            msg += f"\nstdout: {stdout.strip()}"
        super().__init__(msg)
        self.argv = argv
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _looks_like_auth_failure(stderr: str | None) -> bool:
    if not stderr:
        return False
    return any(hint in stderr for hint in _AUTH_HINTS)


# Terminal SLURM job states — when job_state returns one of these, polling can stop.
TERMINAL_JOB_STATES = frozenset(
    {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
     "BOOT_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "DEADLINE"}
)
