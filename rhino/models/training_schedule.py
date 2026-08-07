"""Deterministic optimizer/contact schedule for RHINO training."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingStage:
    optimizer_mode: str
    enable_contact_loss: bool


def get_training_stage(
    epoch: int,
    num_epochs: int,
    contact_weight: float,
    opt_pose_freq: int,
    opt_pose_len: int,
) -> TrainingStage:
    """Return the complete training state for an epoch, including resumes."""
    if epoch < 0:
        raise ValueError(f"epoch must be non-negative, got {epoch}")
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {num_epochs}")

    if contact_weight <= 0:
        mode = "normal" if epoch >= int(0.2 * num_epochs) else "no_pose"
        return TrainingStage(mode, False)

    if opt_pose_freq <= 0:
        raise ValueError(f"opt_pose_freq must be positive, got {opt_pose_freq}")
    if not 0 < opt_pose_len < opt_pose_freq:
        raise ValueError(
            "opt_pose_len must satisfy 0 < opt_pose_len < opt_pose_freq, got "
            f"{opt_pose_len} and {opt_pose_freq}"
        )

    if epoch <= 0.15 * num_epochs:
        return TrainingStage("no_pose", False)
    if epoch % opt_pose_freq >= opt_pose_len:
        return TrainingStage("normal", False)

    mode = "pose" if epoch > 0.5 * num_epochs else "pose_no_hand"
    return TrainingStage(mode, True)
