# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Stereo 3D Object Detection module for YOLO.

This module implements stereo-based 3D object detection with CenterNet-style outputs,
including geometric construction, dense alignment, and occlusion handling based on
the Stereo CenterNet paper.

Key Components:
    - Stereo3DDetModel: Main model class for stereo 3D detection
    - Stereo3DDetTrainer: Training logic with stereo-specific augmentation
    - Stereo3DDetValidator: Validation with KITTI AP3D metrics
    - Stereo3DDetPredictor: Inference pipeline for stereo images

Implementation Modules:
    - nms: Heatmap-based NMS for CenterNet detection
    - keypoints: Perspective-aware keypoint selection
    - geometric: Geometric construction for 3D box estimation
    - dense_align: Sub-pixel stereo refinement via dense alignment
    - occlusion: Occlusion classification and handling

See Also:
    - scripts/benchmark_stereo3ddet.py for benchmarking utilities
"""

# =============================================================================
# Core Module Exports
# =============================================================================
# =============================================================================
# Data Augmentation
# =============================================================================
from .augment import (
    HorizontalFlipAugmentor,
    PhotometricAugmentor,
    RandomCropAugmentor,
    RandomScaleAugmentor,
    StereoAugmentationPipeline,
    StereoCalibration,
)

# =============================================================================
# GAP-002: Dense Alignment (User Story 5)
# =============================================================================
from .dense_align import (
    DenseAlignment,
    create_dense_alignment_from_config,
)

# =============================================================================
# GAP-001: Geometric Construction (User Story 2)
# =============================================================================
from .geometric import (
    CalibParams,
    GeometricConstruction,
    GeometricObservations,
    fallback_simple_triangulation,
    solve_geometric_batch,
    solve_geometric_single,
)

# =============================================================================
# GAP-004: Perspective Keypoint Selection (User Story 1)
# =============================================================================
from .keypoints import (
    get_quadrant_name,
    get_visible_face_indices,
    select_perspective_keypoints,
    select_perspective_keypoints_batch,
)
from .metrics import Stereo3DDetMetrics
from .model import Stereo3DDetModel

# =============================================================================
# GAP-003: Heatmap NMS (User Story 3)
# =============================================================================
from .nms import heatmap_nms

# =============================================================================
# GAP-006: Occlusion Classification (User Story 6)
# =============================================================================
from .occlusion import (
    classify_occlusion,
    get_occlusion_stats,
    should_skip_dense_alignment,
)
from .predict import Stereo3DDetPredictor
from .train import Stereo3DDetTrainer

# =============================================================================
# Utility Functions
# =============================================================================
from .utils import (
    filter_and_remap_class_id,
    get_paper_class_mapping,
    get_paper_class_names,
    is_paper_class,
)
from .val import Stereo3DDetValidator
from .visualize import plot_stereo_predictions, plot_stereo_sample

__all__ = [
    "CalibParams",
    # -------------------------------------------------------------------------
    # GAP-002: Dense Alignment
    # -------------------------------------------------------------------------
    "DenseAlignment",
    # -------------------------------------------------------------------------
    # GAP-001: Geometric Construction
    # -------------------------------------------------------------------------
    "GeometricConstruction",
    "GeometricObservations",
    "HorizontalFlipAugmentor",
    "PhotometricAugmentor",
    "RandomCropAugmentor",
    "RandomScaleAugmentor",
    "Stereo3DDetMetrics",
    # -------------------------------------------------------------------------
    # Core Classes
    # -------------------------------------------------------------------------
    "Stereo3DDetModel",
    "Stereo3DDetPredictor",
    "Stereo3DDetTrainer",
    "Stereo3DDetValidator",
    "StereoAugmentationPipeline",
    # -------------------------------------------------------------------------
    # Data Augmentation
    # -------------------------------------------------------------------------
    "StereoCalibration",
    # -------------------------------------------------------------------------
    # GAP-006: Occlusion Classification
    # -------------------------------------------------------------------------
    "classify_occlusion",
    "create_dense_alignment_from_config",
    "fallback_simple_triangulation",
    "filter_and_remap_class_id",
    "get_occlusion_stats",
    # -------------------------------------------------------------------------
    # Utility Functions
    # -------------------------------------------------------------------------
    "get_paper_class_mapping",
    "get_paper_class_names",
    "get_quadrant_name",
    "get_visible_face_indices",
    # -------------------------------------------------------------------------
    # GAP-003: Heatmap NMS
    # -------------------------------------------------------------------------
    "heatmap_nms",
    "is_paper_class",
    "plot_stereo_predictions",
    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    "plot_stereo_sample",
    # -------------------------------------------------------------------------
    # GAP-004: Perspective Keypoint Selection
    # -------------------------------------------------------------------------
    "select_perspective_keypoints",
    "select_perspective_keypoints_batch",
    "should_skip_dense_alignment",
    "solve_geometric_batch",
    "solve_geometric_single",
]
