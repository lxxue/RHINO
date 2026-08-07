"""Centralized tool configuration for RHINO preprocessing pipeline.

All external tool paths, Python interpreters, and checkpoint locations are
managed through the ToolConfig dataclass. Each preprocessing step imports
this module and validates required tools before execution.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ToolConfig:
    """Configuration for all external preprocessing tools."""

    # Third-party submodule roots (relative to repo root by default)
    interactvlm_root: Path = _REPO_ROOT / "third_party" / "InteractVLM"
    aios_root: Path = _REPO_ROOT / "third_party" / "AiOS"
    sapiens_root: Path = _REPO_ROOT / "third_party" / "sapiens"
    metric3d_root: Path = _REPO_ROOT / "third_party" / "Metric3D"
    sam2_root: Path = _REPO_ROOT / "third_party" / "Grounded-SAM-2"
    mast3r_root: Path = _REPO_ROOT / "third_party" / "mast3r"
    hloc_root: Path = _REPO_ROOT / "third_party" / "Hierarchical-Localization"

    # Model checkpoints
    smplx_model_dir: Path = _REPO_ROOT / "rhino" / "checkpoints" / "smplx"
    sam2_checkpoint: Path = _REPO_ROOT / "third_party" / "Grounded-SAM-2" / "checkpoints" / "sam2_hiera_large.pt"
    """SAM2 model checkpoint path."""
    sapiens_checkpoint_root: Optional[Path] = None
    """Root directory for Sapiens checkpoints (detector + pose). If None, searches sapiens_root."""

    # Python interpreter paths (for tools needing separate conda envs)
    sam2_python: str = "python"
    mast3r_python: str = "python"
    aios_python: str = "python"
    sapiens_python: str = "python"
    interactvlm_python: str = "python"
    metric3d_python: str = "python"


def check_submodule(path: Path, name: str) -> None:
    """Raise RuntimeError if a submodule directory is missing or empty."""
    if not path.is_dir():
        raise RuntimeError(
            f"Submodule '{name}' not found at {path}. "
            f"Run: git submodule update --init --recursive"
        )
    # Check it's not an empty directory (uninitialised submodule)
    if not any(path.iterdir()):
        raise RuntimeError(
            f"Submodule '{name}' at {path} appears empty (not initialised). "
            f"Run: git submodule update --init --recursive"
        )


def require_tools(cfg: ToolConfig, *tool_names: str) -> None:
    """Validate that the requested tools are available.

    Args:
        cfg: The tool configuration to validate against.
        *tool_names: Tool identifiers to check. Supported values:
            interactvlm, aios, sapiens, metric3d, sam2, mast3r, hloc, smplx_model
    """
    validators = {
        "interactvlm": (cfg.interactvlm_root, "InteractVLM"),
        "aios": (cfg.aios_root, "AiOS"),
        "sapiens": (cfg.sapiens_root, "sapiens"),
        "metric3d": (cfg.metric3d_root, "Metric3D"),
        "sam2": (cfg.sam2_root, "Grounded-SAM-2"),
        "mast3r": (cfg.mast3r_root, "mast3r"),
        "hloc": (cfg.hloc_root, "Hierarchical-Localization"),
    }
    for name in tool_names:
        if name == "smplx_model":
            if not cfg.smplx_model_dir.is_dir():
                raise RuntimeError(
                    f"SMPL-X model directory not found at {cfg.smplx_model_dir}. "
                    f"Download SMPL-X models and place them there."
                )
            continue
        if name not in validators:
            raise ValueError(f"Unknown tool name: {name}")
        path, label = validators[name]
        check_submodule(path, label)
