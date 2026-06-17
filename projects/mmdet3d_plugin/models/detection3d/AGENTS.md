## OVERVIEW

3D object detection + tracking via Sparse4DHead. 900 DETR-style learnable anchors, 6 decoder layers with deformable attention, denoising training. Outputs 3D bounding boxes + velocity + instance IDs.

## FILES

| File | Lines | Key Class | Registry |
|------|-------|-----------|----------|
| `detection3d_head.py` | 558 | Sparse4DHead | @HEADS |
| `detection3d_blocks.py` | 300 | SparseBox3DEncoder, SparseBox3DRefinementModule, SparseBox3DKeyPointsGenerator | @POSITIONAL_ENCODING, @PLUGIN_LAYERS |
| `decoder.py` | 107 | SparseBox3DDecoder | @BBOX_CODERS |
| `target.py` | 437 | SparseBox3DTarget | @BBOX_SAMPLERS |
| `losses.py` | 93 | SparseBox3DLoss | @LOSSES |

## DECODER SEQUENCE (operation_order in config)

```
Layer 1 (single-frame):   gnn -> norm -> deformable -> ffn -> norm -> refine
Layers 2-5 (temporal):    temp_gnn -> gnn -> norm -> deformable -> ffn -> norm -> refine
Layer 6:                  gnn -> norm -> deformable -> ffn -> norm -> refine
```

Each operation in detail:

- `gnn`: self-attention among 900 queries (MultiheadFlashAttention)
- `deformable`: cross-attention to multi-view image features (DeformableFeatureAggregation)
- `temp_gnn`: cross-attention from 900 current queries to 600 historical cached instances
- `ffn`: AsymmetricFFN (256 -> 1024 -> 256)
- `refine`: SparseBox3DRefinementModule predicts cls + box deltas + centerness + yawness

## DN TRAINING

Denoising training accelerates convergence. `BaseTargetWithDenoising.sample()` adds noise to GT boxes to create DN queries. DN queries skip InstanceBank and participate in self-attention only. Attention masks prevent DN queries from seeing each other (independent denoising). DN queries are removed before loss computation.

## INSTANCEBANK INTERACTION

```
forward():
  1. anchor_projection() -- compensate ego-motion: prev-frame anchors to current frame
  2. InstanceBank.get() -- returns [current_900_anchors, cached_600_features]
  3. Layer 1: no temp_gnn (no temporal data yet)
  4. After layer 1: InstanceBank.update(confidence_topk=300)
     -> merged = cat([cached_600, current_top300]) -- position inheritance tracking
  5. Layers 2-5: temp_gnn cross-attends to cached_600
  6. get_instance_id() -- IDs for positions [0:599] inherited from previous frame
```

## OUTPUT ENCODING (SparseBox3DDecoder)

11-dim box: X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ. Z, W, L, H decoded with exp(). SIN_YAW and COS_YAW compose to YAW = atan2(). Centerness and yawness are auxiliary quality scores.

## ANTI-PATTERNS

- DO NOT change operation_order without updating num_decoder config
- The 1st layer intentionally skips temp_gnn -- InstanceBank cache is populated after layer 1
