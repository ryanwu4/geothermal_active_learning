"""Wrap ``cli_surrogate_array_prepare.jl`` to stage IX submission for an iteration.

The Julia script consumes a manifest JSON, materializes per-task ``.jl`` files
under a ``stage_run_dir``, and writes a ``submit_surrogate_array.sbatch``
script. We invoke it without ``--remote-userhost`` so it skips the local→scp
helper and gives us a sbatch we can submit immediately.

Hybrid mode (local driver on bend, IX on Sherlock) injects a ``runner`` that
executes the Julia subprocess on Sherlock over ssh — in that case every path
passed in must already be a Sherlock-side absolute path.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


_SBATCH_PATH_RE = re.compile(r"Sbatch file\s*:\s*(\S+)")


class StageError(RuntimeError):
    pass


@dataclass
class StageResult:
    stage_run_dir: Path
    sbatch_path: Path
    tasks_json: Path
    tasks_count: int


# Runner signature: takes an argv list, returns a CompletedProcess-like object
# with .returncode/.stdout/.stderr. Default runs locally; hybrid passes a
# remote runner that ssh's into Sherlock.
Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


def _local_runner(argv: list[str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def stage_iteration(
    *,
    julia_repo: Path,
    julia_config: str,
    surrogate_repo: Path,
    manifest_path: Path,
    stage_root: Path,
    output_prefix: str,
    cpus: int = 2,
    mem: str = "8GB",
    time_limit: str = "00:30:00",
    np_procs: int = 2,
    max_concurrent: int = 50,
    job_name: str = "AL_IX_ARRAY",
    runner: Runner | None = None,
    skip_script_existence_check: bool = False,
) -> StageResult:
    """Run the staging script and parse its stdout for the rendered sbatch path.

    ``runner`` defaults to a local ``subprocess.run``; pass a remote runner for
    hybrid mode and set ``skip_script_existence_check=True`` (the script lives
    on the remote host and we can't stat it locally).
    """
    script = Path(julia_repo) / "scripts" / "cli_surrogate_array_prepare.jl"
    if not skip_script_existence_check and not script.exists():
        raise StageError(f"Staging script not found: {script}")

    cmd = [
        "julia", f"--project={julia_repo}",
        str(script),
        "--manifest", str(manifest_path),
        "--surrogate-root", str(surrogate_repo),
        "--config", str(Path(julia_repo) / julia_config) if not Path(julia_config).is_absolute() else julia_config,
        "--stage-root", str(stage_root),
        "--output-prefix", output_prefix,
        "--cpus", str(cpus),
        "--mem", mem,
        "--time", time_limit,
        "--np", str(np_procs),
        "--max-concurrent", str(max_concurrent),
        "--job-name", job_name,
        "--repo-dir-for-sbatch", str(julia_repo),
    ]

    proc = (runner or _local_runner)(cmd)
    if proc.returncode != 0:
        raise StageError(
            f"Julia stage script failed (code {proc.returncode}):\n"
            f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )

    sbatch_path: Path | None = None
    stage_run_dir: Path | None = None
    tasks_count: int = 0
    for line in proc.stdout.splitlines():
        m = _SBATCH_PATH_RE.search(line)
        if m:
            sbatch_path = Path(m.group(1))
        if "Stage dir" in line:
            stage_run_dir = Path(line.split(":", 1)[1].strip())
        if "Task count" in line:
            try:
                tasks_count = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass

    if sbatch_path is None or stage_run_dir is None:
        raise StageError(
            f"Could not parse stage output. STDOUT was:\n{proc.stdout}"
        )

    tasks_json = stage_run_dir / "manifests" / "array_tasks.json"
    return StageResult(
        stage_run_dir=stage_run_dir,
        sbatch_path=sbatch_path,
        tasks_json=tasks_json,
        tasks_count=tasks_count,
    )
