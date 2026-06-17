# mmdet3d_plugin — SparseDrive Custom Code

## OVERVIEW

This is the mmdetection3d plugin directory: all custom model code, datasets, and ops live here. Booted via `importlib.import_module()` triggered by config's `plugin=True` + `plugin_dir = "projects/mmdet3d_plugin/"`.

## REGISTRATION CHAIN

```
projects/mmdet3d_plugin/__init__.py
  → from .datasets import *     # pipelines, samplers, evaluators, NuScenes3DDataset
  → from .models import *       # detectors, heads, blocks, losses, all task modules
  → from .apis import *         # custom train_detector / test wrappers
  → from .core.evaluation import *  # CustomDistEvalHook
```

That single import triggers ALL `@XX.register_module()` decorators across 27+ classes using 10+ mmdet/mmcv registries.

## REGISTRIES USED

| Registry | Source | Count | Examples |
|---|---|---|---|
| DETECTORS | mmdet | 1 | `SparseDrive` |
| HEADS | mmdet | 3 | `SparseDriveHead`, `Sparse4DHead`, `MotionPlanningHead` |
| BBOX_CODERS | mmdet | 4 | `SparseBox3DDecoder`, `SparsePoint3DDecoder`, etc. |
| BBOX_SAMPLERS | mmdet | 4 | `SparseBox3DTarget`, `MotionTarget`, etc. |
| BBOX_ASSIGNERS | mmdet | 1 | `HungarianLinesAssigner` |
| LOSSES | mmdet | 3 | `SparseBox3DLoss`, `SparseLineLoss`, `LinesL1Loss` |
| MATCH_COST | mmdet | 2 | `LinesL1Cost`, `MapQueriesCost` |
| PIPELINES | mmdet | 11 | All data transforms |
| DATASETS | mmdet | 1 | `NuScenes3DDataset` |
| ATTENTION | mmcv | 2 | `MultiheadFlashAttention`, `DeformableFeatureAggregation` |
| PLUGIN_LAYERS | mmcv | 10 | `InstanceBank`, `InstanceQueue`, `DenseDepthNet`, all `RefinementModule`s |
| POSITIONAL_ENCODING | mmcv | 2 | `SparseBox3DEncoder`, `SparsePoint3DEncoder` |
| FEEDFORWARD_NETWORK | mmcv | 1 | `AsymmetricFFN` |

## SHARED CONSTANTS

`core/box3d.py` — imported by 9 files across detection3d/, motion/, and datasets/:

```python
X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))
CNS, YNS = 0, 1   # centerness and yawness indices in quality
YAW = 6            # decoded
```

**DO NOT** modify these indices without updating all 9 importers.

## DIRECTORY MAP

| Dir | Purpose | Has `__init__.py`? |
|---|---|---|
| `apis/` | Custom train/test wrappers (bypass mmdet defaults) | YES |
| `core/` | `box3d.py` constants + `evaluation/` eval hook | NO (only `evaluation/` has one) |
| `datasets/` | NuScenes3DDataset, pipelines, samplers, 5-task evaluators | YES |
| `models/` | SparseDrive + heads + blocks; subdirs: detection3d/, map/, motion/ | YES |
| `ops/` | Custom CUDA `deformable_aggregation` (must compile before use) | YES |

## ANTI-PATTERNS

- **DO NOT** remove wildcard imports from `__init__.py` — they trigger all registry registration
- **DO NOT** import any plugin module before mmdet `build_detector()` is called (registrations must complete first)
- `core/` has NO `__init__.py` — only `box3d.py` and the `core/evaluation/` subpackage are directly importable
- Configs are standalone flat dicts — never use `_base_` inheritance in this project
