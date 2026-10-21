"""Runtime configuration for the TrackEverything ComfyUI nodes.

Everything is resolved at runtime and can be overridden, so the package never
hard-codes a particular user account or installation path.

Resolution order (first match wins):

``project_root``
    1. ``TRACKEVERYTHING_PROJECT_ROOT`` environment variable
    2. ``<package>/../config.json`` -> ``"project_root"``
    3. the parent directory of this package (the normal checkout layout)

``worker_python``
    1. ``TRACKEVERYTHING_WORKER_PYTHON`` environment variable
    2. ``<package>/../config.json`` -> ``"worker_python"``
    3. ``<project_root>/.pixi/envs/default/bin/python``

``output_root``
    1. ``TRACKEVERYTHING_OUTPUT_ROOT`` environment variable
    2. ``<package>/../config.json`` -> ``"output_root"``
    3. ComfyUI's own output directory (``folder_paths.get_output_directory()``)
    4. ``<project_root>/outputs/comfy``

No setting in this module imports anything from the isolated environment: the
ComfyUI process must stay free of ``trackeverything``, ``torch_scatter`` and
the Python 3.10 packages of the worker environment.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

PACKAGE_DIR = Path(__file__).resolve().parent

ENV_PROJECT_ROOT = "TRACKEVERYTHING_PROJECT_ROOT"
ENV_WORKER_PYTHON = "TRACKEVERYTHING_WORKER_PYTHON"
ENV_OUTPUT_ROOT = "TRACKEVERYTHING_OUTPUT_ROOT"

CONFIG_FILENAME = "config.json"

#: Packages that must never be imported inside the ComfyUI process.
FORBIDDEN_IMPORTS = ("trackeverything", "torch_scatter")


class ConfigurationError(RuntimeError):
    """Raised when the node cannot be configured for the current machine."""


def _read_config_file(directory: Path) -> dict:
    path = directory / CONFIG_FILENAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"{path} must contain a JSON object")
    return data


def _default_project_root() -> Path:
    """The checkout root: the directory that contains this package."""
    return PACKAGE_DIR.parent


def _comfy_output_directory() -> Optional[Path]:
    """ComfyUI's output directory, when running inside ComfyUI."""
    try:
        import folder_paths  # type: ignore

        return Path(folder_paths.get_output_directory())
    except Exception:
        return None


@dataclass
class Config:
    """Resolved configuration for one TrackEverything worker invocation."""

    project_root: Path
    worker_python: Path
    output_root: Path
    upstream_dir: Path
    worker_script: Path
    model_repo: str = "katefgroup/TrackEverything"
    pi3_checkpoint: str = "yyfz233/Pi3"
    device: str = "cuda"
    geometry_backend: str = "pi3"
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def resolve(cls) -> "Config":
        project_root = Path(
            os.environ.get(ENV_PROJECT_ROOT) or _default_project_root()
        ).expanduser().resolve()

        file_cfg = _read_config_file(project_root)

        worker_python = os.environ.get(ENV_WORKER_PYTHON) or file_cfg.get("worker_python")
        if not worker_python:
            worker_python = project_root / ".pixi" / "envs" / "default" / "bin" / "python"

        output_root = os.environ.get(ENV_OUTPUT_ROOT) or file_cfg.get("output_root")
        if not output_root:
            comfy_out = _comfy_output_directory()
            output_root = comfy_out if comfy_out else project_root / "outputs" / "comfy"

        return cls(
            project_root=project_root,
            worker_python=Path(worker_python).expanduser(),
            output_root=Path(output_root).expanduser(),
            upstream_dir=project_root / "external" / "trackeverything",
            worker_script=project_root / "worker" / "runner.py",
            model_repo=os.environ.get("TRACKEVERYTHING_MODEL", file_cfg.get("model_repo", cls.model_repo)),
            pi3_checkpoint=os.environ.get(
                "TRACKEVERYTHING_PI3_CHECKPOINT", file_cfg.get("pi3_checkpoint", cls.pi3_checkpoint)
            ),
            device=os.environ.get("TRACKEVERYTHING_DEVICE", file_cfg.get("device", cls.device)),
            geometry_backend=file_cfg.get("geometry_backend", cls.geometry_backend),
            extra=file_cfg,
        )

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Raise :class:`ConfigurationError` with an actionable message."""
        problems: list[str] = []

        if not self.project_root.is_dir():
            problems.append(
                f"project root {self.project_root} does not exist "
                f"(set ${ENV_PROJECT_ROOT})"
            )

        if not self.worker_python.is_file():
            problems.append(
                f"worker python {self.worker_python} does not exist "
                f"(set ${ENV_WORKER_PYTHON} to the python of the isolated environment)"
            )
        elif not os.access(self.worker_python, os.X_OK):
            problems.append(f"worker python {self.worker_python} is not executable")

        if not self.worker_script.is_file():
            problems.append(f"worker script {self.worker_script} is missing")

        if not self.upstream_dir.is_dir():
            problems.append(
                f"upstream TrackEverything checkout {self.upstream_dir} is missing "
                f"(git submodule update --init --recursive)"
            )

        if problems:
            raise ConfigurationError(
                "TrackEverything nodes are not configured:\n  - " + "\n  - ".join(problems)
            )

    # ------------------------------------------------------------------
    @property
    def upstream_parent(self) -> Path:
        """Parent directory of the upstream package (must be on ``PYTHONPATH``)."""
        return self.upstream_dir.parent

    def describe(self) -> dict:
        return {
            "project_root": str(self.project_root),
            "worker_python": str(self.worker_python),
            "worker_script": str(self.worker_script),
            "upstream_dir": str(self.upstream_dir),
            "output_root": str(self.output_root),
            "model_repo": self.model_repo,
            "pi3_checkpoint": self.pi3_checkpoint,
            "device": self.device,
            "geometry_backend": self.geometry_backend,
        }


# ---------------------------------------------------------------------------
# isolation guard
# ---------------------------------------------------------------------------
def assert_isolated(config: Optional[Config] = None) -> None:
    """Fail loudly if the ComfyUI process leaked into the worker environment.

    The ComfyUI process must never import ``trackeverything`` /
    ``torch_scatter``, and must never see the worker's ``site-packages`` on
    ``sys.path``. This check runs before every worker launch.

    The ``sys.path`` part is skipped when the current interpreter *is* the
    worker interpreter - that is the case for the test harness, which runs
    inside the isolated environment on purpose.
    """
    problems: list[str] = []

    for name in FORBIDDEN_IMPORTS:
        module = sys.modules.get(name)
        if module is not None:
            problems.append(
                f"module '{name}' is already imported in the ComfyUI process "
                f"({getattr(module, '__file__', '?')})"
            )

    if config is not None and config.worker_python.exists():
        same_interpreter = False
        try:
            same_interpreter = (
                Path(sys.executable).resolve() == config.worker_python.resolve()
            )
        except OSError:
            pass

        if not same_interpreter:
            isolated_site = str(config.worker_python.parent.parent / "lib")
            for entry in sys.path:
                if entry and isolated_site in str(entry):
                    problems.append(
                        f"the worker environment's site-packages is on sys.path: {entry}"
                    )

    if problems:
        raise RuntimeError(
            "TrackEverything isolation violated:\n  - " + "\n  - ".join(problems)
        )
