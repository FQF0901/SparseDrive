# models/ — SparseDrive Model Code

## OVERVIEW

All SparseDrive model code. Organized vertically by task (detection3d, map, motion) rather than horizontally by component type. Each task subdirectory is self-contained with its own head, blocks, decoder, target, and loss.

## ARCHITECTURE

```
SparseDrive (@DETECTORS)
  ├── img_backbone: ResNet50 (from mmdet)
  ├── img_neck: FPN, 4 scales → embed_dims=256
  ├── depth_branch: DenseDepthNet (auxiliary depth supervision)
  └── head: SparseDriveHead (@HEADS)
        ├── det_head: Sparse4DHead → detection + tracking
        │     ├── InstanceBank: 900 learnable anchors + 600 temporal cached
        │     ├── 6 decoder layers: gnn → norm → deformable → ffn → norm → refine
        │     └── DN training: denoising anchors for faster convergence
        ├── map_head: Sparse4DHead → online mapping
        │     ├── InstanceBank: 100 point anchors + 33 temporal
        │     └── HungarianLinesAssigner: vector line element matching
        └── motion_plan_head: MotionPlanningHead → prediction + planning
              ├── InstanceQueue: 4-frame FIFO temporal queue with explicit ID matching
              ├── Parallel decoder: 3 layers of (temp_gnn → gnn → cross_gnn → ffn) → refine
              └── Collision-aware rescore via HierarchicalPlanningDecoder
```

## SHARED COMPONENTS (at models/ level)

| File | Key classes | Purpose |
|------|-------------|---------|
| `blocks.py` | DeformableFeatureAggregation (`@ATTENTION`), DenseDepthNet, AsymmetricFFN (`@FEEDFORWARD_NETWORK`) | Cross-view multi-scale attention, depth estimation, asymmetric feed-forward |
| `attention.py` | MultiheadFlashAttention (`@ATTENTION`) | FlashAttention 2.3.2 wrapper |
| `instance_bank.py` | InstanceBank (`@PLUGIN_LAYERS`) | DETR-style learnable query bank with temporal caching |
| `base_target.py` | BaseTargetWithDenoising | Abstract base for all target samplers, provides DN logic |
| `grid_mask.py` | GridMask | Applied in `forward()` rather than pipeline (unconventional) |

## TASK SUBDIRECTORY CONVENTIONS

Each task subdir follows a consistent internal structure:

| File | detection3d/ | map/ | motion/ |
|------|-------------|------|---------|
| `*_head.py` | Sparse4DHead | Sparse4DHead (~same class, reused) | MotionPlanningHead |
| `*_blocks.py` | RefinementModule, KeypointGenerator | RefinementModule, KeypointGenerator | MotionRefinementModule, MotionKeypointGenerator |
| `decoder.py` | SparseBox3DDecoder | SparsePoint3DDecoder | PlanningDecoder + HierarchicalPlanningDecoder |
| `target.py` | SparseBox3DTarget (DN) | SparsePoint3DTarget (mask) | MotionTarget + PlanningTarget |
| extra | losses.py | loss.py + match_cost.py | instance_queue.py |

## NAMING PREFIXES

- Detection: `SparseBox3D*` — boxes in 3D space
- Mapping: `SparsePoint3D*` — line elements as point sequences
- Motion: `Motion*`, `Planning*`, `SparseBox3DMotion*` — mixed, least consistent of the three

## ANTI-PATTERNS

- **DO NOT** assume `.blocks` imports are consistent — detection3d and map import `from ..blocks import linear_relu_ln`, but motion does NOT (it imports only `from ..blocks import linear_relu_ln` is absent).
- **DO NOT** assume type annotations — map/ has full typing, motion/ has almost none, detection3d/ is mixed.
- **DO NOT** add new files to `models/` without adding a wildcard import in `__init__.py` — registration will silently fail.
- The map head reuses the same `Sparse4DHead` class as detection, configured differently (100 anchors vs 900, point-based vs box-based). They are NOT separate classes.
