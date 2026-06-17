# PROJECT KNOWLEDGE BASE

**Generated:** 2025-06-17
**Commit:** ec0225d
**Branch:** fqf_branch

## OVERVIEW

SparseDrive — end-to-end autonomous driving via sparse scene representation. mmdetection3d plugin-based model for detection, tracking, online mapping, motion prediction, and planning on nuScenes. Sparse-centric paradigm unifying 5 tasks with symmetric sparse perception + parallel motion planner.

**Stack:** Python 3.8, PyTorch 1.13 + CUDA 11.6, mmcv-full 1.7.1, mmdet 2.28.2, FlashAttention 2.3.2

## STRUCTURE

```
SparseDrive/
├── projects/
│   ├── configs/                          # 2 flat configs (stage1/stage2), no _base_ inheritance
│   └── mmdet3d_plugin/                   # ALL custom code — the real project
│       ├── __init__.py                   # Plugin boot: imports datasets, models, apis, core.evaluation
│       ├── apis/                         # Custom train/test wrappers (bypass mmdet's train_detector)
│       ├── core/
│       │   ├── box3d.py                 # 3 lines: axis indices X..VZ = range(11), imported everywhere
│       │   └── evaluation/              # CustomDistEvalHook (eval loop, not metrics)
│       ├── datasets/
│       │   ├── nuscenes_3d_dataset.py   # 1123 lines — central dataset + 5-task eval dispatch
│       │   ├── pipelines/               # Data transforms (@PIPELINES registered)
│       │   ├── samplers/                # Sequence-aware batch samplers
│       │   └── evaluation/              # Actual METRICS live here (map/motion/planning/)
│       ├── models/
│       │   ├── sparsedrive.py           # @DETECTORS: SparseDrive (ResNet50+FPN+DepthNet → SparseDriveHead)
│       │   ├── sparsedrive_head.py      # @HEADS: dispatches to det_head, map_head, motion_plan_head
│       │   ├── blocks.py               # Shared: DeformableFeatureAggregation, DenseDepthNet, AsymmetricFFN
│       │   ├── attention.py            # MultiheadFlashAttention wrapper (flash-attn 2.3.2)
│       │   ├── instance_bank.py         # DETR-style learnable query bank (detection memory)
│       │   ├── base_target.py           # BaseTargetWithDenoising (all target samplers extend this)
│       │   ├── grid_mask.py            # GridMask applied in forward (not pipeline) — unusual
│       │   ├── detection3d/            # Sparse4DHead — 6 decoder layers, 900 anchors, DN training
│       │   ├── map/                    # SparsePoint3D* — 100 anchors, vector line matching
│       │   └── motion/                 # MotionPlanningHead — parallel motion(12f)+planning(6f)
│       └── ops/                        # Custom CUDA: deformable_aggregation (must compile before use)
├── tools/                              # train.py, test.py, benchmark.py, kmeans/, data_converter/
├── scripts/                            # Shell wrappers: train.sh, test.sh, create_data.sh, kmeans.sh
├── docs/
│   ├── quick_start.md                  # Setup + training instructions
│   └── architecture.md                 # 1021-line Chinese architecture deep-dive
├── requirement.txt                      # ⚠️ singular (not requirements.txt), all == pinned
└── .gitignore                          # Excludes: data/, ckpt/, work_dirs*/
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| **Plugin registration** | `projects/mmdet3d_plugin/__init__.py` → chains through sub-`__init__.py` | All registrations triggered by single `importlib.import_module("projects.mmdet3d_plugin")` |
| **Model forward pass** | `projects/mmdet3d_plugin/models/sparsedrive.py` | Backbone+Neck+DepthNet → SparseDriveHead |
| **Sparse4DHead (det)** | `projects/mmdet3d_plugin/models/detection3d/detection3d_head.py` | 558 lines — temporal instances, DN, 6-layer decoder |
| **Map head** | `projects/mmdet3d_plugin/models/map/map_blocks.py` | 100 point anchors, HungarianLinesAssigner |
| **Motion+planning head** | `projects/mmdet3d_plugin/models/motion/motion_planning_head.py` | Parallel decoder, collision-aware rescore |
| **Deformable attention** | `projects/mmdet3d_plugin/models/blocks.py` + `ops/` | Custom CUDA kernel (not mmcv MSAttn) |
| **Instance tracking** | `projects/mmdet3d_plugin/models/instance_bank.py` | Position-based ID inheritance (no explicit match) |
| **Dataset + eval** | `projects/mmdet3d_plugin/datasets/nuscenes_3d_dataset.py` | NuScenes3DDataset.evaluate() dispatches 5 task evaluators |
| **Config schema** | `projects/configs/sparsedrive_small_stage2.py` | Flat dict — all model/dataset/opt/schedule inline |
| **Training entry** | `tools/train.py` + `scripts/train.sh` | 2-stage: stage1(det+map 100ep) → stage2(+motion+plan 10ep) |

## CONVENTIONS

**Configs:**
- Flat Python dicts — NO `_base_` inheritance. Each config fully self-contained.
- `type` keys → mmdet/mmcv registries for component instantiation.
- `plugin = True` + `plugin_dir = "projects/mmdet3d_plugin/"` triggers dynamic import.
- Config variables: snake_case, `dict(key=VAR)` pattern, double-quoted keys.
- 2-stage training: stage2 loads stage1 via `load_from = 'ckpt/sparsedrive_stage1.pth'`.

**Naming:**
- `Sparse-` prefix for SparseDrive-specific classes: `SparseDrive`, `Sparse4DHead`, `SparseBox3D*`, `SparsePoint3D*`
- Suffix conventions: `*Head`, `*Decoder`, `*Target`, `*Loss`, `*Encoder`, `*Generator`, `*RefinementModule`
- Task distinction: `SparseBox3D*` (detection) vs `SparsePoint3D*` (mapping)
- File naming: snake_case, mirrors class names (e.g., `instance_bank.py` → `InstanceBank`)

**Code:**
- Google-style docstrings, `__all__` in most `__init__.py`
- Registry decorators on ALL registered classes: `@DETECTORS.register_module()`, `@HEADS.register_module()`, `@PLUGIN_LAYERS.register_module()`, etc.
- Wildcard imports in `__init__.py`: `from .datasets import *`, `from .models import *`
- Import `from projects.mmdet3d_plugin.core.box3d import *` for box axis indices
- Double quotes for strings, 4-space indent, no enforced line length
- Type annotations: inconsistent — map/ has full typing, motion/ has almost none

**10 Registry types used:**
`DETECTORS`, `HEADS`, `BBOX_CODERS`, `BBOX_SAMPLERS`, `BBOX_ASSIGNERS`, `LOSSES`, `MATCH_COST`, `PIPELINES`, `DATASETS` (from mmdet) + `ATTENTION`, `PLUGIN_LAYERS`, `POSITIONAL_ENCODING`, `FEEDFORWARD_NETWORK` (from mmcv) + custom `SAMPLER`, `OBJECTSAMPLERS`

## ANTI-PATTERNS (THIS PROJECT)

- **DO NOT** use `_base_` config inheritance — configs are standalone flat dicts
- **DO NOT** change `core/box3d.py` axis indices without updating all importers (7+ files `from ..core.box3d import *`)
- **DO NOT** remove `from .X import *` in `__init__.py` — these trigger registry registration
- **DO NOT** import project files before `build_detector()` is called — registrations must complete first
- **DO NOT** skip CUDA op compilation (`projects/mmdet3d_plugin/ops/setup.py develop`) — deformable attention will crash at runtime
- **DO NOT** use `torchrun` — `tools/dist_train.sh` uses deprecated `torch.distributed.launch`
- **NEVER** skip the k-means anchor generation step — model loads anchors from `data/kmeans/*.npy`

## UNIQUE STYLES

- **Plugin architecture**: All custom code in `projects/mmdet3d_plugin/`, bootstrapped via `importlib.import_module()`. Not a pip-installable package.
- **Task-vertical model organization**: `detection3d/`, `map/`, `motion/` — each self-contained with decoder/blocks/target/loss. Contrast with mmdet3d's horizontal organization (detectors/necks/heads/).
- **`requirement.txt`** (singular) instead of `requirements.txt`
- **Metrics split**: eval *loop* in `core/evaluation/`; actual metric *implementations* in `datasets/evaluation/{map,motion,planning}/`
- **`grid_mask.py` as model component**: GridMask applied in forward pass, NOT via pipeline transform
- **No test suite** — zero unit/integration tests. Validation = benchmark numbers only.
- **No CI/CD** — no GitHub Actions, no linting, no pre-commit hooks

## COMMANDS

```bash
# Prerequisites
pip install -r requirement.txt
cd projects/mmdet3d_plugin/ops && python setup.py develop && cd ../../..

# Data prep + anchors
sh scripts/create_data.sh          # → data/infos/*.pkl
sh scripts/kmeans.sh               # → data/kmeans/*.npy

# 2-stage training
sh scripts/train.sh                # or: tools/dist_train.sh projects/configs/sparsedrive_small_stage1.py 8
                                   # then: tools/dist_train.sh projects/configs/sparsedrive_small_stage2.py 8

# Testing
sh scripts/test.sh                 # or: tools/dist_test.sh projects/configs/sparsedrive_small_stage2.py ckpt/sparsedrive_stage2.pth 8

# Benchmark
python tools/benchmark.py projects/configs/sparsedrive_small_stage2.py --checkpoint ckpt/sparsedrive_stage2.pth

# Visualization
python tools/visualization/visualize.py projects/configs/sparsedrive_small_stage2.py --result-path results.pkl
```

## NOTES

- **Huge architecture doc exists**: `docs/architecture.md` (1021 lines, Chinese) — the most detailed reference for model internals. Mine this before reading source.
- **InstanceBank tracking is position-based** (concat order), NOT explicit ID matching. InstanceQueue uses explicit matching. See `architecture.md` §11.
- **DeformableFeatureAggregation**: 6 views × 4 scales × 7 keypoints = 168 positions share ONE softmax (unified cross-view attention). Custom CUDA kernel.
- **Two known "ugly workarounds"**: `tools/train.py:230` (det/seg model detection) and `apis/mmdet_train.py:137` (log filename matching). Consider fixing if modifying training infra.
- **5 TODOs exist** — see agent scan in init-deep Phase 1.
- **Anchor counts**: detection 900 (DETR-style dense), map 100 (sparse lines), motion 6 modes, planning 6 modes
- **This project does NOT import `mmdetection3d`**. All 3D logic (projections, box encoding, keypoint sampling) implemented in-plugin.
