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

SERVER_METHODS = {
    "fedavg",
    "fedprox",
    "fedper",
    "fedrep",
    "ditto",
}

DECENTRALIZED_REFERENCE_METHODS = {
    "dpsgd_full_neighbors",
}

DESCRIPTOR_METHODS = {
    "prototype_only",
    "quality_only",
    "prototype_quality",
    "prototype_quality_exploration",
    "anchor_quality",
    "hard_class_alignment",
    "pearl_full",
    "model_similarity",
}

SUPPORTED_METHODS = (
    set(DEFAULT_METHODS)
    | SERVER_METHODS
    | DECENTRALIZED_REFERENCE_METHODS
    | {"dpsgd_one_peer", "model_similarity"}
)

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
    "dpsgd_one_peer": "#2F6B5F",
    "dpsgd_full_neighbors": "#76B7B2",
    "model_similarity": "#B07AA1",
    "fedavg": "#4E79A7",
    "fedprox": "#E15759",
    "fedper": "#59A14F",
    "fedrep": "#F28E2B",
    "ditto": "#EDC948",
}

METHOD_WIDTHS = {
    method: 2.5 if method == "pearl_full" else 1.5
    for method in METHOD_COLORS
}
