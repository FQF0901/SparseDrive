## OVERVIEW

NuScenes data loading, augmentation pipelines, sequence-aware samplers, and 5-task evaluation metrics. Central dataset class is NuScenes3DDataset (1123 lines -- largest file in the project).

## FILES

| File | Lines | Purpose |
|------|-------|---------|
| `nuscenes_3d_dataset.py` | 1123 | NuScenes3DDataset: data loading + `__getitem__` + `evaluate()` dispatch to 5 subtasks |
| `builder.py` | 192 | `custom_build_dataset()`, `build_dataloader()`, OBJECTSAMPLERS registry |
| `utils.py` | 225 | Shared visualization: `draw_lidar_bbox3d_on_img` / `draw_lidar_bbox3d_on_bev` |
| `pipelines/` | 5 files | Transforms: augment (233), loading (188), transform (242), vectorize (207) |
| `samplers/` | 5 files | DistributedGroupSampler, GroupInBatchSampler -- sequence-aware batch construction |
| `evaluation/` | 7 files | 5-task metrics: det/track (mmdet3d), map (AP/distance/vector_eval), motion (uniad-style), planning (collision+L2) |
| `map_utils/` | 2 files | nuScenes Map API wrappers (no `__init__.py` -- flat utilities) |

## EVALUATION DISPATCH

`NuScenes3DDataset.evaluate()` routes to 5 sub-evaluators based on `eval_mode` config:

```
eval_mode = dict(
    with_det=True,      => _evaluate_single()  -- mmdet3d DetectionMetrics
    with_tracking=True,  => _evaluate_single()  -- AMOTA, AMOTP
    with_map=True,       => VectorEvaluate()    -- AP, chamfer distance
    with_motion=True,    => NuScenesEvalMotion() -- EPA, minADE, minFDE, miss rate
    with_planning=True,  => planning_eval()     -- L2 displacement, collision rate
)
```

Each sub-evaluator lives under `evaluation/{map,motion,planning}/` as flat utility files (no `__init__.py` in subdirectories). The `evaluate()` method itself is 100+ lines of dispatch logic.

## DATA PIPELINE (train_pipeline order)

1. `LoadMultiViewImageFromFiles` -- reads 6 RGB images `[6, 3, H, W]`
2. `LoadPointsFromFile` -- LiDAR (for GT boxes, not model input)
3. `ResizeCropFlipImage` -- resize to (704, 256), random flip
4. `MultiScaleDepthMapGenerator` -- GT depth maps at 3 scales
5. `BBoxRotation` -- 3D box rotation augmentation
6. `PhotoMetricDistortionMultiViewImage` -- color jitter
7. `NormalizeMultiviewImage` -- ImageNet mean/std
8. `CircleObjectRangeFilter` -- drops objects beyond 55m
9. `InstanceNameFilter` -- filters to 10 class names
10. `VectorizeMap` -- map polygons to point sequences
11. `NuScenesSparse4DAdaptor` -- projection matrices, timestamps for Sparse4D
12. `Collect` -- gathers all keys for model input

## ANTI-PATTERNS

- Evaluation metrics live in `datasets/evaluation/`, **not** `core/evaluation/` (hooks are in core, metric implementations are here).
- `evaluation/map/`, `evaluation/motion/`, `evaluation/planning/` have **no** `__init__.py` -- imported by path, not as packages.
- `map_utils/` also has no `__init__.py` -- same flat-utility pattern.
- `nuscenes_3d_dataset.py` is a monolith: `evaluate()` alone is 100+ lines with 5 sub-evaluator calls inlined.
- `evaluation/__init__.py` exists but is empty (0 lines) -- it's a namespace marker only.
