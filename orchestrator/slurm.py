"""SLURM helpers — sbatch submission with dependency capture and template rendering.

Two patterns we use throughout the chain:

* :func:`submit_sbatch` — submit an sbatch script and return the captured job
  id, optionally with a ``--dependency=<spec>:<job_id>`` constraint and extra
  ``--export=K=V`` env vars threaded into the rendered script.
* :func:`render_template` — substitute ``{{KEY}}`` tokens inside an sbatch
  template before writing it to the per-iteration sbatch dir. Templates are
  intentionally simple — Python ``str.replace`` rather than Jinja2 to avoid an
  extra dependency on the conda env.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Mapping


_JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")


class SbatchError(RuntimeError):
    pass


def render_template(template_path: Path | str, substitutions: Mapping[str, str]) -> str:
    """Render a sbatch template with ``{{KEY}}`` tokens replaced from substitutions.

    Missing keys are left as-is (so the rendered file fails loudly if invoked)
    rather than silently producing garbage.
    """
    text = Path(template_path).read_text()
    for k, v in substitutions.items():
        text = text.replace(f"{{{{{k}}}}}", str(v))
    return text


def write_rendered(template_path: Path | str, dest_path: Path | str, substitutions: Mapping[str, str]) -> Path:
    rendered = render_template(template_path, substitutions)
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(rendered)
    dest.chmod(0o755)
    return dest


def submit_sbatch(
    script_path: Path | str,
    *,
    dependency: str | None = None,
    extra_args: list[str] | None = None,
    cwd: Path | str | None = None,
) -> str:
    """Submit ``script_path`` via sbatch and return the SLURM job id.

    ``dependency`` is a full spec like ``afterok:12345`` or
    ``afterany:12345:12346``; the caller is responsible for picking the right
    form.
    """
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(str(script_path))

    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SbatchError(
            f"sbatch failed with code {proc.returncode}: stderr={proc.stderr!r} stdout={proc.stdout!r}"
        )

    # With --parsable, sbatch prints just the numeric job id (sometimes
    # ``<id>;<cluster>``). Strip whitespace and a trailing cluster suffix.
    raw = proc.stdout.strip().split(";")[0]
    if not raw.isdigit():
        # Fall back to parsing the older "Submitted batch job 12345" format.
        m = _JOB_ID_RE.search(proc.stdout)
        if not m:
            raise SbatchError(f"Could not parse sbatch job id from output: {proc.stdout!r}")
        raw = m.group(1)
    return raw


def parse_array_job_id(stdout: str) -> str:
    """Parse a job id from an sbatch stdout that wasn't run with --parsable.

    Handles both ``Submitted batch job 12345`` and bare numeric forms. Useful
    when invoking the Julia staging script which submits its own sbatch.
    """
    text = stdout.strip()
    if text.isdigit():
        return text
    m = _JOB_ID_RE.search(text)
    if not m:
        raise SbatchError(f"Could not find job id in: {stdout!r}")
    return m.group(1)
