"""Frame-selection contracts shared by datasets and training modules."""


def build_frame_indices(
    start_frame: int,
    end_frame: int,
    skip_step: int,
    exclude_frames: list[int] | None = None,
    *,
    available_count: int | None = None,
) -> list[int]:
    """Build and validate the source-frame indices used by a dataset split."""
    if start_frame < 0:
        raise ValueError(f"start_frame must be non-negative, got {start_frame}")
    if end_frame < start_frame:
        raise ValueError(
            f"end_frame must be >= start_frame, got {end_frame} < {start_frame}"
        )
    if skip_step <= 0:
        raise ValueError(f"skip_step must be positive, got {skip_step}")

    indices = list(range(start_frame, end_frame + 1, skip_step))
    selected = set(indices)
    excluded = set(exclude_frames or [])
    invalid_exclusions = sorted(excluded - selected)
    if invalid_exclusions:
        raise ValueError(
            "exclude_frames contains frames outside the selected range/cadence: "
            f"{invalid_exclusions}"
        )

    indices = [idx for idx in indices if idx not in excluded]
    if not indices:
        raise ValueError("Frame selection is empty after applying exclude_frames")

    if available_count is not None:
        if available_count <= 0:
            raise ValueError("No source frames are available")
        if indices[-1] >= available_count:
            raise ValueError(
                f"Selected frame {indices[-1]} exceeds the available frame range "
                f"[0, {available_count - 1}]"
            )
    return indices
