# SparseDrive vs SparseWorld: 完整架构对比分析

> 基于 SparseDrive 代码仓库 + ICRA 2025 paper vs SparseWorld arxiv:2605.24394 (May 2026, ZJU + Huawei)
> SparseWorld 没有公开代码仓库 — 本文档旨在梳理差异并为后续实现提供路线图

---

## 目录

1. [核心定位差异](#1-核心定位差异)
2. [整体架构对比](#2-整体架构对比)
3. [模块级差异详解](#3-模块级差异详解)
4. [数据流对比](#4-数据流对比)
5. [训练方式对比](#5-训练方式对比)
6. [推理方式对比](#6-推理方式对比)
7. [监督信号与损失函数对比](#7-监督信号与损失函数对比)
8. [从 SparseDrive 到 SparseWorld 的实现路线图](#8-从-sparsedrive-到-sparseworld-的实现路线图)
9. [开放问题与潜在风险](#9-开放问题与潜在风险)

---

## 1. 核心定位差异

| 维度 | SparseDrive | SparseWorld |
|------|------------|-------------|
| **Paper 目标** | 提出 Sparse-Centric 范式，替代 BEV-Centric | 提出轻量级 World Model 插件，增强 E2E planning |
| **核心创新** | 对称稀疏感知 + 并行运动规划器 | 自回归未来实例预测 + 基于未来预测的规划精炼 |
| **本质关系** | **基础架构 (baseline)** | **附加模块 (plug-in)** — 建立在 SparseDrive 之上 |
| **兼容性** | 独立系统 | 兼容 SparseDrive 和 VAD 两种 baseline |
| **表征粒度** | 稀疏实例 (agent + map) | 复用同样的稀疏实例表征 |
| **时间建模** | 历史帧 → 当前帧 (InstanceBank/Queue) | 当前帧 → 未来帧 (自回归 rollout) |

**关键认知**: SparseWorld **不是** SparseDrive 的替代品。它是 SparseDrive 上的一个 **世界模型增强模块**，类似给 SparseDrive 装了一个"模拟器"来预测未来场景。

---

## 2. 整体架构对比

### 2.1 SparseDrive 架构 (当前代码)

```
多视图图像 [B,6,3,H,W]
    │
    ▼
ResNet50 Backbone → FPN Neck
    │ 4 scales, [B,6,256,Hi,Wi]
    ▼
┌─────────────────────────────────────────┐
│       SparseDriveHead                    │
│                                          │
│  det_head (Sparse4DHead)                 │
│  ├── InstanceBank: 900 anchors + 600 temporal │
│  ├── 6 decoder layers:                   │
│  │   gnn → norm → deformable → ffn → refine │
│  │   (+ temp_gnn for temporal layers)    │
│  └── Output: 3D boxes + velocity + ID   │
│                                          │
│  map_head (Sparse4DHead)                 │
│  ├── InstanceBank: 100 anchors + 33 temporal │
│  ├── Same decoder structure              │
│  └── Output: vector map elements        │
│                                          │
│  motion_plan_head (MotionPlanningHead)   │
│  ├── InstanceQueue: 4-frame FIFO         │
│  ├── 3 decoder layers:                   │
│  │   temp_gnn → gnn → cross_gnn → ffn   │
│  ├── Motion prediction: 6 modes × 12 ts  │
│  └── Planning: 6 modes × 6 ts            │
│      └── HierarchicalPlanningDecoder     │
│          ├── Collision-aware rescore     │
│          └── Max score selection         │
└─────────────────────────────────────────┘
    │
    ▼
Loss: L_det + L_map + L_motion + L_plan + L_depth
```

### 2.2 SparseWorld 架构 (从 paper 还原)

```
多视图图像 [B,6,3,H,W]
    │
    ▼
ResNet50 Backbone → FPN Neck (与 SparseDrive 完全相同)
    │
    ▼
┌─────────────────────────────────────────┐
│    Instance-Aware Driving Baseline       │  ← 即 SparseDrive 的感知 + 初始运动规划
│    (= SparseDrive without CAR to output) │
│    Output:                               │
│    - Current instances I0 (agent+map)    │
│    - Initial motion trajectories         │
│    - Action conditions C0                │
└──────────────┬──────────────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌──────────────────┐  ┌──────────────────────┐
│  Sparse Dreamer  │  │  Future Motion       │
│  (NEW)           │◄─┤  Planner (FMP, NEW)  │
│                  │  │  = copy of baseline   │
│  GIA: anchor预投影│  │  motion planner       │
│  SWD: N transformer│  │                      │
│   decoders:       │  │  Input: future inst.  │
│   self-attn       │  │  Output: future action│
│   temp-cross-attn │──┤  conditions Ct+1      │
│   action-cross-attn│  │                      │
│   FFN + refine    │  │                      │
│                  │  │                      │
│  Input: IMQ中的   │  │                      │
│  历史实例+action │  │                      │
│  Output: It+1    │  │                      │
└──────────────────┘  └──────────────────────┘
               │                 │
               └──── autoregressive rollout ──┘
               (重复 f 次: It → FMP → Ct+1 → SWD → It+2)

               │
               ▼
┌─────────────────────────────────────────┐
│    Motion Planning Refinement (NEW)      │
│                                          │
│  Motion Prediction Refinement:           │
│    用预测的未来实例 {I1..If} 替代历史实例  │
│    → 重新运行 baseline motion predictor   │
│    → refined agent trajectories          │
│                                          │
│  Trajectory Planning Refinement:         │
│    ego feature cross-attn with future     │
│    instance features → refined ego traj. │
│    + Safety-Critical Loss (SCL, NEW)     │
│                                          │
│  Adaptive Trajectory Selection (ATS, NEW)│
│    3 candidates: baseline / future / refined│
│    → collision check + SCL → select safest│
│    → apply adjustment vector             │
└─────────────────────────────────────────┘
    │
    ▼
最终输出: 最安全的规划轨迹
```

---

## 3. 模块级差异详解

### 3.1 共享模块 (SparseWorld 完全复用 SparseDrive)

| 模块 | SparseDrive 代码位置 | 在 SparseWorld 中的使用 |
|------|---------------------|------------------------|
| **Backbone (ResNet50)** | `sparsedrive.py` → mmdet ResNet | 完全复用 |
| **Neck (FPN)** | `sparsedrive.py` → mmdet FPN | 完全复用 |
| **DenseDepthNet** | `blocks.py` | 完全复用 (辅助监督) |
| **Sparse4DHead (检测)** | `detection3d/detection3d_head.py` | 完全复用 (Instance Perception 阶段) |
| **Sparse4DHead (地图)** | 同上，不同参数 | 完全复用 |
| **InstanceBank** | `instance_bank.py` | 完全复用 |
| **DeformableFeatureAggregation** | `blocks.py` + `ops/` | 完全复用 |
| **MultiheadFlashAttention** | `attention.py` | 完全复用 |
| **AsymmetricFFN** | `blocks.py` | 完全复用 |
| **Det head decoder** | `detection3d/decoder.py` | 完全复用 |
| **Map head decoder** | `map/decoder.py` | 完全复用 |

### 3.2 新增模块 (SparseWorld 独有)

| 模块 | 功能 | 复杂度估计 | 建议文件名 |
|------|------|-----------|-----------|
| **Sparse Dreamer** | 自回归未来实例预测 | ~300-500 行 | `models/motion/sparse_dreamer.py` |
| ├─ Global Instance Alignment (GIA) | anchor 预投影 (ego-motion + velocity) | ~50 行 | 内含于 above |
| └─ SparseWorld Decoder (SWD) | N 个 transformer decoder | ~200 行 | 内含于 above |
| **Future Motion Planner (FMP)** | 从未来实例推断 action conditions | ~100 行 | 可复用 motion_planning_head 逻辑 |
| **Safety-Critical Loss (SCL)** | 基于未来预测的碰撞约束 | ~100 行 | `models/motion/safety_loss.py` |
| **Adaptive Trajectory Selection (ATS)** | 三选一最安全轨迹 | ~80 行 | `models/motion/adaptive_trajectory_selection.py` |
| **Motion Planning Refinement** | 用未来实例精炼 motion + planning | ~150 行 | 集成到 motion_planning_head 中 |

### 3.3 修改模块 (SparseWorld 对 SparseDrive 做了修改)

| 模块 | 修改内容 | 影响范围 |
|------|---------|---------|
| **InstanceQueue** | 原来只存历史实例(4帧)，现在也存预测的未来实例 | `instance_queue.py` |
| **MotionPlanningHead** | 增加 world model forwarding + refinement 分支 | `motion_planning_head.py` |
| **Loss 计算** | 增加 SCL loss，增加未来预测的检测/地图 loss | config 中的 loss weights |
| **Config** | 新增 SparseDreamer 配置项，新增 FMP 配置项 | `sparsedrive_small_stage2.py` + 新 config |

---

## 4. 数据流对比

### 4.1 SparseDrive 数据流 (单帧)

```
当前帧图像 I_t
  → Backbone+Neck → Feature Maps F_t
  → det_head(F_t, InstanceBank)    → detection boxes + instance_id
  → map_head(F_t, InstanceBank)    → vector map elements
  → motion_plan_head(
      det_output_top50,
      map_output_top10,
      InstanceQueue(history_4frames)
    )
    → 6 motion modes × 12 timesteps
    → 6 planning modes × 6 timesteps
    → HierarchicalPlanningDecoder(collision rescore)
    → 1 final trajectory
```

### 4.2 SparseWorld 数据流 (多帧 + 自回归)

```
当前帧图像 I_t + 历史帧 I_{t-h..t-1}
  → Instance Perception (同 SparseDrive)
  → 当前实例 I_0 = {A_0, F_0}  (agents + map)
  → 初始运动规划 (同 SparseDrive)
  → 初始 action conditions C_0 = (v_0, τ_0, s_0)
  → 初始 ego trajectory τ^base

  ┌─ Autoregressive Rollout Loop (for k = 0..f-1) ─┐
  │                                                    │
  │  FMP: C_k = FMP(I_k)  →  velocity + traj + cmd   │
  │  GIA: A'_k = project(A_k, ego_comp, velocity)    │
  │  SWD: I_{k+1} = SWD(I_{k-m..k}, C_k, A'_k)      │
  │       ├─ self-attn (current instances)            │
  │       ├─ temp-cross-attn (historical instances)   │
  │       ├─ action-cross-attn (C_k embedding)        │
  │       ├─ FFN                                      │
  │       └─ classification + box/polyline refine     │
  │                                                    │
  └────────────────────────────────────────────────────┘

  → 未来实例序列 {I_1, I_2, ..., I_f}  (agent forecasts + map forecasts)

  ┌─ Motion Planning Refinement ─────────────────────┐
  │                                                    │
  │  Motion Refinement:                                │
  │    motion_predictor(I_0, {I_1..I_f}) → refined     │
  │    agent trajectories τ^ref_agent                   │
  │                                                    │
  │  Planning Refinement:                              │
  │    ego_feature = cross_attn(F_e, {F_1..F_f})       │
  │    planning_head(ego_feature) → τ^ref_ego          │
  │    SCL: v_adj = compute_adjustment(τ, agents_future)│
  │                                                    │
  │  ATS:                                             │
  │    candidates = {τ^base, τ^future, τ^refined}      │
  │    for each: collision_check + SCL                │
  │    select safest with lowest SCL, no collision    │
  │    apply adjustment vector                        │
  │    → final trajectory τ*                           │
  └────────────────────────────────────────────────────┘
```

---

## 5. 训练方式对比

### 5.1 SparseDrive 训练

| 阶段 | 内容 | Epochs | 可训练模块 |
|------|------|--------|-----------|
| Stage 1 | det + map | 100 | Backbone, Neck, DepthNet, DetHead, MapHead |
| Stage 2 | det + map + motion + plan | 10 | 全部模块 (stage1权重加载后继续训练) |

**关键特性**:
- 2阶段训练，Stage 2 加载 Stage 1 权重
- 每帧独立处理 (无自回归 rollout)
- 所有 loss 可单帧计算

### 5.2 SparseWorld 训练 (从 Paper 推断)

SparseWorld 的训练方式 paper 没有明确描述细节，但从架构可以推断：

**可能方案 A: 三阶段训练**
| 阶段 | 内容 | 新增模块 |
|------|------|---------|
| Stage 1 | 同 SparseDrive Stage 1 | — |
| Stage 2 | 同 SparseDrive Stage 2 (初始运动规划器) | — |
| Stage 3 | Sparse Dreamer + Refinement (freeze baseline?) | SPD, FMP, Refinement, SCL, ATS |

**可能方案 B: 端到端联合训练**
- 在 SparseDrive Stage 2 的基础上直接加入 SparseDreamer + Refinement
- 所有模块联合训练
- 需要 teacher forcing 来训练自回归模块

**需要确定的关键训练细节:**
1. FMP 的权重是否与 baseline motion planner 共享？(paper 说 "identical network architecture" — 可能共享)
2. Sparse Dreamer 的训练是否需要 teacher forcing？
3. SCL 的训练是否需要在推理时也用调整向量 (adjustment vector)？
4. 未来预测 (mAP, NDS) 的 GT 从哪来？— 从 nuScenes 的未来帧标注

### 5.3 训练 Loss 对比

| Loss | SparseDrive | SparseWorld |
|------|:----------:|:-----------:|
| L_det (cls + reg) | ✓ (Focal + L1) | ✓ (复用 baseline) |
| L_map (cls + reg) | ✓ (Focal + LinesL1) | ✓ (复用 baseline) |
| L_depth (aux) | ✓ (L1, w=0.2) | ✓ (复用 baseline) |
| L_motion (cls + reg) | ✓ (Focal, w=0.2 | L1, w=0.2) | ✓ (复用 baseline) |
| L_plan (cls + reg + status) | ✓ (Focal, w=0.5 | L1, w=1.0 | L1, w=1.0) | ✓ (复用 baseline) |
| **L_future_det** | ✗ | **NEW**: SparseDreamer 预测的未来检测 loss |
| **L_future_map** | ✗ | **NEW**: SparseDreamer 预测的未来地图 loss |
| **L_scl (Safety-Critical)** | ✗ | **NEW**: 基于未来 agent 预测的碰撞避免 loss |
| **L_refined_motion** | ✗ | **NEW**: 精炼后的 motion prediction loss |
| **L_refined_plan** | ✗ | **NEW**: 精炼后的 planning loss |

---

## 6. 推理方式对比

### 6.1 SparseDrive 推理

```
单次 forward pass:
  images → perception → motion_plan → trajectory
  耗时: ~111ms (9 FPS on 4090)
  内存: ~1.3GB

输出: 1 条最终轨迹 (通过 collision-aware rescore 选择)
```

### 6.2 SparseWorld 推理

```
多阶段推理:
  1. image → perception → I_0, C_0, τ^base      (~74ms, baseline部分)
  2. autoregressive rollout (f=4 steps):
     for k=0..3: FMP(I_k) → C_k → SWD → I_{k+1}  (~70ms, 4帧)
  3. refinement:
     motion_refine + plan_refine + ATS             (~30ms)
  总耗时: ~174ms (~5.7 FPS on 4090)
  内存: ~4.4GB

输出: 3 条候选轨迹中选 1 条最安全轨迹

候选轨迹来源:
  - Baseline trajectory (τ^base)
  - World model predicted future trajectory (τ^future)
  - Safety-Critical Loss refined trajectory (τ^refined)

选择机制 (ATS):
  1. 对每条候选用 future agent predictions 做碰撞检测
  2. 计算 SCL
  3. 选 SCL 最低且无碰撞的轨迹
  4. 若都安全，优先选 τ^refined (精炼后)
  5. 若有碰撞，应用 adjustment vector v_adj
```

---

## 7. 监督信号与 GT 对比

### 7.1 SparseDrive 需要的 GT

| GT 数据 | Shape | 用途 |
|---------|-------|------|
| gt_bboxes_3d | [N, 9] | 检测 loss |
| gt_labels_3d | [N] | 检测分类 loss |
| gt_depth | [6, 3, H, W] | 深度辅助 loss |
| gt_map_pts | [M, 20, 2] | 地图 loss |
| gt_map_labels | [M] | 地图分类 loss |
| gt_agent_fut_trajs | [N, 12, 2] | motion prediction loss |
| gt_agent_fut_masks | [N, 12] | motion 有效帧 mask |
| gt_ego_fut_trajs | [6, 2] | planning loss |
| gt_ego_fut_masks | [6] | planning 有效帧 mask |
| gt_ego_fut_cmd | [1] | 驾驶指令 (turn/go) |

### 7.2 SparseWorld 额外需要的 GT

| 新增 GT 数据 | Shape | 来源 | 用途 |
|-------------|-------|------|------|
| **未来帧检测 GT** | [N_future, 9] | nuScenes 未来帧标注 (已有，需额外提取) | 训练 SparseDreamer 的 agent 预测 |
| **未来帧地图 GT** | [M_future, 20, 2] | nuScenes 未来帧 map (可能从未来帧的 ego pose + 当前 map 投影) | 训练 SparseDreamer 的 map 预测 |
| **未来帧 action conditions** | (v, τ, s) × f steps | 可从 GT ego trajectory 推导 | 训练 FMP |
| **碰撞 GT (可选)** | — | 从未来帧的 agent box 和 ego box 计算 | SCL 的监督信号 |

**GT 提取策略**:
- **ngents**: nuScenes 每帧都有未来 6s (12 frames @ 2Hz) 的 agent 轨迹 GT，可用于构造未来帧的 agent anchors
- **Map**: map elements 是静态的，但需要根据 ego-motion 变换到未来帧的坐标系。可从当前帧的 map GT + 未来 ego pose 计算
- **Action conditions**: 从 GT ego trajectory 可以反向计算 velocity, trajectory, steering

---

## 8. 从 SparseDrive 到 SparseWorld 的实现路线图

### Phase 1: 基础架构复用确认 (分析阶段，不写代码)

- [x] 确认 SparseDrive 所有可复用模块
- [ ] 确认哪些模块需要修改 (InstanceQueue, MotionPlanningHead)
- [ ] 确认哪些模块完全新增 (SparseDreamer, SCL, ATS)

### Phase 2: 数据准备

- [ ] 修改 `nuscenes_converter.py` — 提取未来帧的 GT 数据
  - 未来帧 agent boxes (从 gt_agent_fut_trajs 反推)
  - 未来帧 map elements (从当前 map + 未来 ego pose 变换)
  - 未来 action conditions GT
- [ ] 修改数据 pipeline (`pipelines/`) — 添加未来帧相关字段

### Phase 3: Sparse Dreamer 实现 (核心)

- [ ] `models/motion/sparse_dreamer.py`:
  - `GlobalInstanceAlignment` 类: anchor 预投影 (ego-motion + velocity compensation)
  - `ActionEmbedding` 类: Fourier embedding of (v, τ, s)
  - `SparseWorldDecoder` 类: N 个 transformer decoder layers
    - 复用 `MultiheadFlashAttention` for self-attn
    - 新增 temporal-cross-attn (use temporal + relative pos embeddings)
    - 新增 action-conditioned cross-attn
    - 复用 `AsymmetricFFN`
    - 复用 `SparseBox3DRefinementModule` for agent refine
    - 新增 map refinement module (复用 SparsePoint3DRefinementModule)
  - `SparseDreamer` 总控类: 组装 GIA + SWD + autoregressive loop
  - `FutureMotionPlanner` 类: 复用 baseline motion planner 逻辑

### Phase 4: Motion Planning Refinement 实现

- [ ] 修改 `models/motion/motion_planning_head.py`:
  - 添加 `forward_world_model()` 方法 — 控制 autoregressive rollout
  - 添加 `refine_motion_prediction()` 方法 — 用未来实例精炼 motion
  - 添加 `refine_trajectory_planning()` 方法 — 用未来实例精炼 planning
- [ ] `models/motion/safety_loss.py`:
  - `SafetyCriticalLoss` 类: 计算 v_adj 和 SCL
  - `compute_adjustment_vector()` 函数: 基于未来 agent 预测的碰撞检测
- [ ] `models/motion/adaptive_trajectory_selection.py`:
  - `AdaptiveTrajectorySelection` 类: 三选一 + collision check + adjustment

### Phase 5: InstanceQueue 修改

- [ ] 修改 `models/motion/instance_queue.py`:
  - 支持存储 future instances (不只是历史)
  - 新增 `push_future()` 方法 (区别于 `push()` for history)
  - 新增 `get_future_instances()` 方法

### Phase 6: Config 和 Loss 更新

- [ ] 新增 config: `sparsedrive_small_stage3_sparseworld.py`
  - 包含 SparseDreamer 配置
  - 包含 FMP 配置
  - 新增 loss weights: λ_future_det, λ_future_map, λ_scl, λ_refined_motion, λ_refined_plan
- [ ] 更新 `sparsedrive_head.py` forward: 可选的 world model 路径

### Phase 7: 训练和验证

- [ ] 训练脚本: 支持 teacher forcing 或 scheduled sampling
- [ ] 推理脚本: 实现 full autoregressive rollout + ATS
- [ ] 验证: 复现 paper 中的 coll. rate 0.05% 和 Bench2Drive DS 48.95

---

## 9. 开放问题与潜在风险

### 9.1 Paper 中未明确的实现细节

| # | 问题 | 重要性 | 推测 |
|---|------|--------|------|
| 1 | SWD 的 N 是多少层？ (decoder 层数) | 高 | 推测 3-6 层，与 Sparse4DHead 的 6 层一致 |
| 2 | 自回归框架共享 baseline motion planner？ | 高 | Paper 说 "identical network architecture" — 可能共享权重 |
| 3 | 训练时 teacher forcing 还是 scheduled sampling？ | 高 | Paper 没提。业界做法: 先 teacher forcing 后 scheduled sampling |
| 4 | SCL 的 θ (safety distance) 是多少？ | 中 | Paper 没提具体值, 需要调参 |
| 5 | 未来预测的 loss weights 是多少？ | 中 | Paper 没提，推测与当前帧 loss 同权重或更低 |
| 6 | FMP 的 action conditions 输出是什么格式？ | 中 | (v, τ, s) = velocity, planned trajectory, steering — 与 SparseDrive 的 ego status 输出格式相同 |
| 7 | 是否需要 warmup SparseDreamer 再 joint training？ | 中 | 推测是: 先单独训练 SparseDreamer (teacher forcing) 再 joint |
| 8 | ATS 中 "future trajectory" 怎么来的？ | 中 | 从 FMP 的 planning 输出得到 (是 world model 预测的 ego 轨迹) |

### 9.2 潜在实现风险

| 风险 | 描述 | 缓解措施 |
|------|------|---------|
| **显存爆炸** | 自回归 rollout f 帧，每帧都要存 instance features，显存可能翻 f 倍 | 使用 gradient checkpointing，或减少 f (paper 评估 2s=4帧) |
| **训练不稳定** | 自回归训练容易出现 error accumulation | Teacher forcing + scheduled sampling |
| **Map 未来预测困难** | Map 是静态的，依赖准确的 ego-motion compensation | GIA 已设计为处理此问题；可实现为 projection baseline + residual |
| **Collision 0% 可能过拟合** | Paper 显示 SparseWorld-S 碰撞率 0.00% (1s/2s)，可能对 nuScenes overfit | 需要在 Bench2Drive (closed-loop) 上也验证 |
| **SCL 的调整向量可能不合理** | 基于 box overlap 的简单碰撞检测可能过于保守或激进 | 可 tune θ (safety distance threshold) |

### 9.3 值得质疑的 Paper 结论

1. **L2 error 反而变差了**: SparseWorld-S 的 L2 Avg 从 0.61 → 0.65 (上升 6.6%)。这是 safety-accuracy tradeoff — 为了安全牺牲了轨迹精度。用户需要确认这个 tradeoff 是否可接受。

2. **FMP "identical network architecture" 的歧义**: 这可能意味着 FMP 是 baseline motion planner 的**共享权重副本**（推理时复用 baseline 训练好的权重），也可能是一个**独立训练的新实例**。前者更优雅但 paper 没说清楚。

3. **Map 预测的价值存疑**: Ablation (Table VIII) 显示单独加 map future 比单独加 agent future 提升更小 (0.619 vs 0.610 EPA)。Map 是静态的，未来预测的价值可能主要是提供 spatial context 而非真正的 temporal forecasting。

---

## 附录 A: SparseWorld 关键配置参数

| 参数 | 值 | 含义 |
|------|-----|------|
| f | 4 | 预测的未来帧数 (2.0s @ 2Hz) |
| m | ? (paper 未明确) | IMQ 中选择的历史实例帧数 |
| N | ? (paper 未明确) | SparseWorld Decoder 层数 |
| c | 256 | 特征维度 (与 SparseDrive 一致) |
| N_d | 900 | Agent anchors |
| N_m | 100 | Map anchors |
| K_m | 6 | Motion modes |
| K_p | 6 | Planning modes |
| T_m | 12 | Motion prediction timesteps |
| T_p | 6 | Planning timesteps |
| θ | ? (paper 未明确) | 安全距离阈值 (用于 SCL) |
| H (IMQ) | 3 (来自 SparseDrive) | 存储帧数 (SparseWorld 扩展为存 history+future) |

---

## 附录 B: 不相关的其他 "SparseWorld" 论文

存在另外两篇同名论文，**与本项目无关**:

1. **SparseWorld (AAAI 2026)**: `arxiv:2510.17482` — Dang et al. (HUST/Tsinghua/Lenovo) — 4D occupancy world model with sparse queries. 使用不同的方法，基于 occupancy 而非 instance。

2. **SparseWorld-TC (CVPR 2026)**: `arxiv:2511.22039` — Du et al. (Tongji/Li Auto) — Trajectory-conditioned sparse occupancy world model. 同样不相关。

我们要实现的是 **arxiv:2605.24394** (May 2026, ZJU + Huawei)。

---

*文档生成时间: 2025-06-17*
*基于: SparseDrive 代码仓库 (ec0225d) + SparseDrive ICRA 2025 Paper + SparseWorld arxiv:2605.24394*
