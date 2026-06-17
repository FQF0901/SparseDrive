## OVERVIEW

Parallel motion prediction (12 future timesteps, 6 modes) and planning (6 future timesteps, 6 modes) via MotionPlanningHead. Uses 4-frame InstanceQueue for temporal context. Collision-aware hierarchical rescoring for safe planning.

## FILES

| File | Lines | Key Class | Registry |
|------|-------|-----------|----------|
| `motion_planning_head.py` | 482 | MotionPlanningHead | @HEADS |
| `motion_blocks.py` | -- | MotionPlanningRefinementModule | @PLUGIN_LAYERS |
| `decoder.py` | -- | SparseBox3DMotionDecoder, HierarchicalPlanningDecoder | @BBOX_CODERS |
| `target.py` | -- | MotionTarget, PlanningTarget | @BBOX_SAMPLERS |
| `instance_queue.py` | -- | InstanceQueue | @PLUGIN_LAYERS |

## PARALLEL DECODER DESIGN

The key innovation: motion prediction and planning run in PARALLEL (not sequential):

```
Input: det_head top-50 detections + map_head top-10 map elements
  -> fused query (concatenated)
  -> 3 decoder layers:
      temp_gnn -> gnn -> cross_gnn -> ffn -> norm
  -> refine layer:
      ├── motion branch: SparseBox3DMotionDecoder -> 6 modes x 12 timesteps x (x,y)
      └── planning branch: HierarchicalPlanningDecoder
            ├── 6 modes x 6 timesteps x (x,y)
            ├── collision check with predicted agent trajectories
            └── rescore: penalize colliding trajectories, select safest
```

- `temp_gnn`: self-attention across the 4-frame temporal queue
- `gnn`: self-attention among motion/planning anchors
- `cross_gnn`: cross-attention between motion anchors and planning anchors
- `ffn`: AsymmetricFFN (256 -> 512 -> 256, half of detection's 1024)

## INSTANCEQUEUE (vs InstanceBank)

| Property | InstanceQueue (motion) | InstanceBank (detection) |
|----------|------------------------|--------------------------|
| Structure | 4-frame FIFO | Single cached frame |
| Matching | Explicit ID matching | Position inheritance (implicit) |
| Content | det_head features + boxes (NO map features) | det_head features only |
| Tracking threshold | 0.2 (in eval config) | Confidence-based topk |

## MOTION TARGET FORMAT

- Output: cumsum trajectory deltas (not absolute positions)
- 6 modes per agent class (car, truck, bus, etc. -- 10 classes)
- MotionTarget selects best mode via Hungarian matching on trajectory distance

## PLANNING TARGET FORMAT

- Output: 6 modes x 6 timesteps x 2D (x, y) + ego status (10-dim)
- 3 commands: turn left, go straight, turn right
- HierarchicalPlanningDecoder: collision check against predicted agent boxes -> rescore trajectories -> select safest

## ANTI-PATTERNS

- DO NOT expect consistent typing -- motion/ has the least type annotations of all 3 task modules
- MotionTarget does NOT inherit BaseTargetWithDenoising (unlike SparseBox3DTarget and SparsePoint3DTarget)
- InstanceQueue does NOT store map features -- only detection features (map is static, no temporal needed)
- motion_blocks.py lacks `__all__` unlike detection3d_blocks.py
