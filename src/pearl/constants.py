"""Shared constants for PEARL experiments."""

DEFAULT_METHODS = [
    "local_only",
    "random_peer",
    "static_peer",
    "prototype_only",
    "quality_only",
    "prototype_quality",
    "prototype_quality_exploration",
    "anchor_quality",
    "hard_class_alignment",
    "pearl_full",
]

METHOD_COLORS = {
    "local_only": "#888780",
    "random_peer": "#378ADD",
    "static_peer": "#BA7517",
    "prototype_only": "#1D9E75",
    "quality_only": "#D4537E",
    "prototype_quality": "#7F77DD",
    "prototype_quality_exploration": "#D85A30",
    "anchor_quality": "#E24B4A",
    "hard_class_alignment": "#0F6E56",
    "pearl_full": "#3C3489",
}

METHOD_WIDTHS = {
    method: 2.5 if method == "pearl_full" else 1.5
    for method in METHOD_COLORS
}
