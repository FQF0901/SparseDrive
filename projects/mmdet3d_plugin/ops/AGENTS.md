# ops — Custom CUDA Extension

## OVERVIEW

Custom CUDA extension for DeformableFeatureAggregation — the core cross-view multi-scale attention operator. Must be compiled before any training. Compiles C++/CUDA source into a PyTorch extension.

## FILES

| File | Purpose |
|------|---------|
| `__init__.py` | Public API: `deformable_aggregation_function()`, `feature_maps_format()` |
| `deformable_aggregation.py` | PyTorch autograd Function wrapper (forward + backward) |
| `setup.py` | Build script: `CUDAExtension('deformable_aggregation_ext')` |
| `src/deformable_aggregation.cpp` | C++ binding (PYBIND11_MODULE) |
| `src/deformable_aggregation_cuda.cu` | CUDA kernel — unified multi-view multi-scale attention |

## COMPILATION

```bash
cd projects/mmdet3d_plugin/ops
python setup.py develop
cd ../../..
```

This registers `deformable_aggregation_ext` as an importable module. The public API functions in `__init__.py` import from this compiled extension. Failure to compile = `ImportError` at runtime.

## PUBLIC API

- `deformable_aggregation_function(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step)` — the core attention operator
- `feature_maps_format(feature_maps)` — reformats multi-level feature maps into the tensor layout expected by the CUDA kernel

## WHAT MAKES IT SPECIAL

Unlike mmcv's `MultiScaleDeformableAttention`, this kernel:
- Operates across 6 camera views × 4 feature scales simultaneously
- 168 keypoint positions (6 views × 4 scales × 7 keypoints per anchor) sharing ONE softmax
- Custom "residual_mode='cat'" for feature aggregation

## ANTI-PATTERNS

- NEVER skip compilation — the model CRASHES at first deformable attention call
- The CUDA kernel expects specific tensor layouts — `feature_maps_format()` must be called first
- This is NOT compatible with standard mmcv `MultiScaleDeformableAttention` — it's a custom operator
