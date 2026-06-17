# SparseDrive 网络架构完全解析

> 基于 `projects/configs/sparsedrive_small_stage2.py` 配置 + 模型源代码追溯
> 框架：mmdetection3d | 骨干：ResNet50 + FPN | 隐层维度：256

---

## 目录

1. [整体数据流](#1-整体数据流)
2. [图像编码器（Backbone + Neck）](#2-图像编码器backbone--neck)
3. [深度估计分支（辅助监督）](#3-深度估计分支辅助监督)
4. [SparseDriveHead 总调度](#4-sparsedrivehead-总调度)
5. [3D 检测头（Sparse4DHead）](#5-3d-检测头sparse4dhead)
6. [地图头（MapHead）](#6-地图头maphead)
7. [运动预测 + 规划头（MotionPlanningHead）](#7-运动预测--规划头motionplanninghead)
8. [InstanceBank —— 检测记忆模块](#8-instancebank--检测记忆模块)
9. [InstanceQueue —— 运动时序队列](#9-instancequeue--运动时序队列)
10. [InstanceBank vs InstanceQueue 完整对比](#10-instancebank-vs-instancequeue-完整对比)
11. [追踪机制是如何工作的](#11-追踪机制是如何工作的)
12. [损失函数总表](#12-损失函数总表)
13. [文本网络结构图](#13-文本网络结构图)

---

## 1. 整体数据流

```
多视图图像 [B, 6, 3, 704, 256]
       │
       ▼
┌─────────────────────────────┐
│     ResNet50 (Backbone)     │  ← strided 4, 8, 16, 32
│     输出 4 层 feature maps   │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│     FPN (Neck)              │  ← 统一到 embed_dims=256
│     输出 4 层 [B,6,256,H,W]  │
└──────────┬──────────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌────────┐  ┌──────────────┐
│ Depth  │  │ Feature Maps │────▶ SparseDriveHead
│ Branch │  │ (4 scales)   │       ├── DetHead (Sparse4DHead)
└────────┘  └──────────────┘       ├── MapHead (Sparse4DHead)
      │                            └── MotionPlanHead (MotionPlanningHead)
      ▼
Depth Loss (aux)
```

### 关于 Feature Maps 的关键说明

Feature Maps 的排列是 `[B, 6, 256, H_i, W_i]`，**不是 24 张独立特征图**。

- `dim=0`：batch
- `dim=1`：6 个相机（cam0~cam5）—— 这**不是**被展平的，而是作为注意力操作的一个维度统一处理
- `dim=2`：channel（256）
- `dim=3,4`：空间维度（随尺度变化）

**DeformableFeatureAggregation 在一个操作内跨全部 6 相机 × 4 尺度做注意力聚合**。对每个 3D 锚点：
1. 生成 7 个 3D keypoints
2. 用 6 个相机的投影矩阵得到 `[B, 900, 7, 6, 2]` 的 2D 投影点
3. 学习的 attention weights shape `[B, 900, 6, 4, 7, 8]` —— 每个锚点为**每个相机 × 每个尺度 × 每个 keypoint** 都学一个权重
4. 加权求和得到最终特征

**不是 24 张图各自独立处理，而是统一的 feature space + 学习的注意力权重自动决定信息流向。**

### 关键维度一览

| 符号 | 含义 | 值/Shape |
|------|------|----------|
| `B` | batch size | 6（8 GPUs × 6 = 48） |
| `num_cams` | 相机数 | 6 |
| `C` | 输入图像通道 | 3 |
| `H, W` | 输入图像尺寸（final_dim） | 256 × 704 |
| `embed_dims` | 统一隐层维度 | 256 |
| `num_levels` | FPN 特征层数 | 4（stride 4,8,16,32） |
| `num_groups` | 注意力头数 / deformable groups | 8 |
| `num_decoder` | decoder layers | 6 |
| `num_anchor` | 检测锚点数 | 900 |
| `num_map_anchor` | 地图锚点数 | 100 |
| `num_classes` | 检测类别数 | 10 |
| `num_map_classes` | 地图类别数 | 3 |
| `queue_length` | 时序队列长度 | 4（history + current） |
| `fut_ts` | 运动预测未来帧数 | 12 |
| `fut_mode` | 运动预测模态数 | 6 |
| `ego_fut_ts` | 自车规划未来帧数 | 6 |
| `ego_fut_mode` | 自车规划模态数 | 6 |
| `num_sample` | 地图线采样点数 | 20 |

---

## 2. 图像编码器（Backbone + Neck）

### 2.1 ResNet50

```python
img_backbone=dict(
    type="ResNet", depth=50,
    frozen_stages=-1,           # 全部可训练
    out_indices=(0,1,2,3),      # 输出 4 层
    with_cp=True,               # gradient checkpointing
    pretrained="ckpt/resnet50-19c8e357.pth",
)
```

| Stage | 输出 Shape（单图） | Stride |
|-------|-------------------|--------|
| layer0 | [B×6, 256, 64, 176] | 4 |
| layer1 | [B×6, 512, 32, 88] | 8 |
| layer2 | [B×6, 1024, 16, 44] | 16 |
| layer3 | [B×6, 2048, 8, 22] | 32 |

### 2.2 FPN

```python
img_neck=dict(
    type="FPN", num_outs=4,
    out_channels=256,            # embed_dims
    in_channels=[256,512,1024,2048],
    add_extra_convs="on_output",
    relu_before_extra_convs=True,
)
```

**功能**：将 ResNet50 的 4 层输出通过 FPN 融合，统一到 256 通道。

**输出**：4-level 多尺度特征图，每个 shape 为 `[B, 6, 256, H_i, W_i]`。

| Level | Stride | H_i | W_i |
|-------|--------|-----|-----|
| 0 | 4 | 64 | 176 |
| 1 | 8 | 32 | 88 |
| 2 | 16 | 16 | 44 |
| 3 | 32 | 8 | 22 |

### 2.3 GridMask（数据增强）

训练时启用，以 `prob=0.7` 随机遮挡图像区域，仅在 backbone 之前应用。

---

## 3. 深度估计分支（辅助监督）

### DenseDepthNet

```python
depth_branch=dict(
    type="DenseDepthNet",
    embed_dims=256,
    num_depth_layers=3,          # 对应前 3 个尺度
    loss_weight=0.2,
)
```

**输入**：FPN 输出的前 3 层特征图 `feature_maps[:3]`（stride 4, 8, 16）

**处理**：
- 每个尺度的特征图通过 1×1 Conv2d(256→1) + exp() 得到深度值
- 深度值 = `exp(conv(feat)) × focal / equal_focal`
- 最后深度值 clamp 到 [0, max_depth=60]

**输出**：3 个尺度的深度图

| 层 | 输入 Shape | 输出 Shape |
|----|-----------|-----------|
| 0 | [B×6, 256, 64, 176] | [B×6, 1, 64, 176] |
| 1 | [B×6, 256, 32, 88] | [B×6, 1, 32, 88] |
| 2 | [B×6, 256, 16, 44] | [B×6, 1, 16, 44] |

**监督信号**：`gt_depth` —— 由数据预处理 `MultiScaleDepthMapGenerator` 从 LiDAR 点云投影生成。

**Loss**: 仅在 foreground pixels（gt_depth > 0）上计算 L1 误差，scale 求和。

```
loss_dense_depth = sum_i(|depth_pred_i - gt_depth_i|) / (N * num_depth_layers) * 0.2
```

---

## 4. SparseDriveHead 总调度

文件：`sparsedrive_head.py`

```python
class SparseDriveHead(BaseModule):
    task_config = dict(with_det=True, with_map=True, with_motion_plan=True)
```

### forward() 数据流

```
Feature Maps [B,6,256,*,*]
       │
       ├──▶ det_head.forward(feature_maps, metas)
       │       └── det_output = {
       │              classification: [List of 6 tensors [B,900,10]],
       │              prediction:     [List of 6 tensors [B,900,11]],
       │              quality:        [List of 6 tensors [B,900,2]],
       │              instance_feature: [B,900,256],
       │              anchor_embed:     [B,900,256],
       │              instance_id:      [B,900],
       │           }
       │
       ├──▶ map_head.forward(feature_maps, metas)
       │       └── map_output = {
       │              classification: [List of 6 tensors [B,100,3]],
       │              prediction:     [List of 6 tensors [B,100,40]],
       │              instance_feature: [B,100,256],
       │              anchor_embed:     [B,100,256],
       │           }
       │
       └──▶ motion_plan_head.forward(
                det_output, map_output, feature_maps, metas,
                det_head.anchor_encoder, det_head.instance_bank.mask,
                det_head.instance_bank.anchor_handler)
                └── motion_output, planning_output
```

### loss() 聚合

```python
losses = {}
losses.update(det_head.loss(det_output, data))      # 检测 Loss
losses.update(map_head.loss(map_output, data))        # 地图 Loss
losses.update(motion_plan_head.loss(motion, plan, data))  # 运动+规划 Loss
```

---

## 5. 3D 检测头（Sparse4DHead）

文件：`detection3d_head.py`

### 5.1 InstanceBank（锚点初始化与时序）

**配置**：`num_anchor=900`，从 `kmeans_det_900.npy` 加载聚类锚点。

每个锚点是一个 11 维向量：`[x, y, z, w, l, h, sin_yaw, cos_yaw, vx, vy, vz]`

- 当前帧：900 个可学习锚点 + 可学习 instance features `[900, 256]`
- 时序帧：保留 top-600 个高置信度实例，通过 `InstanceBank.cache()` 维护

**InstanceBank.get()** 返回：
- `instance_feature`: `[B, 900, 256]`（当前帧）
- `anchor`: `[B, 900, 11]`（当前帧）
- `temp_instance_feature`: `[B, 600, 256]`（时序帧，可能 None）
- `temp_anchor`: `[B, 600, 11]`（时序帧）
- `time_interval`: `[B]`（时间差）

**时序补偿**：时序锚点通过 `anchor_projection()` 用 `T_global` / `T_global_inv` 矩阵做坐标系对齐。

### 5.2 SparseBox3DEncoder（锚点编码）

将锚点的不同分量分别编码并融合：

```
pos_fc(xyz)   → [128]
size_fc(wlh)  → [32]
yaw_fc(sin_yaw,cos_yaw) → [32]
vel_fc(vx,vy,vz) → [64]
output = cat([pos_feat, size_feat, yaw_feat, vel_feat])  → [256]
```

输出：`anchor_embed: [B, 900, 256]`

### 5.3 Decoder Layer 操作序列

总共 6 个 decoder layers，各层的操作序列为（以 stage2 配置为例）：

```python
# Layer 1 (single-frame):
  ["gnn", "norm", "deformable", "ffn", "norm", "refine"]
# Layers 2-6 (temporal):
  ["temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine"]
```

注意 stage2 的 `operation_order` 从第 2 个元素开始切片 `[2:]`，所以实际为：
```
Layer 0: deformable, ffn, norm, refine       (single-frame 剩余)
Layer 1: temp_gnn, gnn, norm, deformable, ffn, norm, refine  (temporal start)
Layer 2-5: same as Layer 1
```

#### 每个操作的作用：

| 操作 | 组件 | 功能 | 输入/输出维度 |
|------|------|------|-------------|
| `temp_gnn` | MultiheadFlashAttention(256×2→256×2→256) | **时序 cross-attention**：当前帧 900 queries 看上帧 600 cached keys/values（不是 self-attention！） | [B,900,256]→[B,900,256] |
| `gnn` | MultiheadFlashAttention(256×2→256×2→256) | 实例间 self-attention：当前帧 900 queries 互相交互 | [B,900,256]→[B,900,256] |
| `norm` | LayerNorm(256) | 层归一化 | [B,900,256]→[B,900,256] |
| `deformable` | DeformableFeatureAggregation | 可变形注意力：从多视图多尺度特征图采样 | [B,900,256]→[B,900,256] |
| `ffn` | AsymmetricFFN(256→1024→256) | 前馈网络 | [B,900,256]→[B,900,256] |
| `refine` | SparseBox3DRefinementModule | 输出分类分数 + 回归 refinement + quality | 详见 5.4 |

### 5.4 SparseBox3DRefinementModule（refine 操作）

```
instance_feature + anchor_embed → MLP → delta
anchor[refine_state] = anchor[refine_state] + delta[refine_state]
velocity: 从位移 / time_interval 计算
```

**输出**：
- `anchor`: `[B, 900, 11]`（更新后的锚点）
- `cls`: `[B, 900, 10]`（分类 logits）
- `quality`: `[B, 900, 2]`（centerness, yawness）

**refine_state**（可 refined 状态）: `[x, y, z, w, l, h, sin_yaw, cos_yaw]`

**velocity**（第 9-11 维）：`(output_delta + anchor) / time_interval`

### 5.5 DeformableFeatureAggregation（可变形特征聚合）—— 跨 6 视图 × 4 尺度的统一注意力

> ⚠️ **关键澄清**：这不是 24 张图分别做 deformable attention 再拼起来。这是一个**统一的跨视图、跨尺度、跨关键点的可变形注意力操作**，6 个视图和 4 个尺度共享同一组 query（anchor feature），通过学习的注意力权重自动决定每个锚点应该关注哪些视图和尺度。

对每个实例锚点（共 900 个）：
1. 通过 `SparseBox3DKeyPointsGenerator` 从 3D anchor box 生成 7 个 keypoints（1 个中心 + 6 个面心偏移），每个 keypoint 是 3D 坐标 `(x,y,z)`
2. 用 6 个相机的 `projection_mat[4×4]` 将 3D keypoints 投影到 6 个视图的 2D 坐标
3. 对每个锚点，学习一个 shape 为 `[6, 4, 7, 8]` 的 attention weight（= 6cams × 4levels × 7pts × 8groups），softmax 归一化在 `cam×level×pts` 维度
4. 用这些权重对 4 个 FPN 尺度 × 6 个视图 × 7 个 keypoints 做**加权采样聚合**，得到最终特征

```
key_points = [B, 900, 7, 3]          # 3D keypoints（从 3D box + offset 生成）
points_2d  = [B, 900, 7, 6, 2]       # 投影到 6 个相机的 2D 坐标
weights    = [B, 900, 6, 4, 7, 8]    # 注意力权重（view, level, pts, groups）
                                       # softmax 在 (cam×level×pts) 维度
features   = sum(view, level, pts, groups weighted) → [B, 900, 256]
```

**等效视角**：
- 每个 anchor 有 7 个 3D 关键点 × 6 个相机 = 42 个 2D 采样位置
- 每个采样位置在 4 个 FPN 尺度上各有一个特征
- 总共 42 × 4 = 168 个候选特征，通过注意力权重加权融合
- 权重由 `instance_feature + anchor_embed + camera_embed` 动态生成

**关键参数**：`num_learnable_pts=6, fix_scale=[7 个固定偏移]`

**关于 "multi-group" vs "multi-head"**：代码中 deformable attention 用 8 groups（`num_groups=8`），每个 group 独立在 168 个采样位置上做 softmax 然后 concat。gNN/temp_gNN 用 8 heads（`num_heads=8`），标准 Q·K^T 注意力。两者都是 256ch → 8×32ch 的拆分，**语义等价**——"groups"只是 deformable attention 中沿用的命名习惯。

### 5.6 目标分配（SparseBox3DTarget）

使用 Hungarian 算法（线性分配）进行 one-to-one matching：

**cost matrix**:
- `cls_cost`: Focal loss style cost（alpha=0.25, gamma=2.0, weight=2.0）
- `box_cost`: L1 cost on encoded box（weight=0.25）

**encoded regression target**:
```
[x, y, z, log(w), log(l), log(h), sin(yaw), cos(yaw), vx, vy, vz]
```

**类别特殊处理**：`traffic_cone` 的 yaw 权重为 0（朝向无意义）

### 5.7 检测头 Loss

```
loss_cls = FocalLoss(sigmoid, gamma=2.0, alpha=0.25, weight=2.0)  # 所有 decoder layers 求和
loss_reg = SparseBox3DLoss:
    loss_box      = L1Loss(weight=0.25)        # box 回归
    loss_cns      = CrossEntropyLoss(sigmoid)   # centerness
    loss_yns      = GaussianFocalLoss           # yawness
```

每个 decoder layer 独立计算 loss，所有层求和。

### 5.8 Detection 解码输出

```python
SparseBox3DDecoder.decode()
# yaw = atan2(sin_yaw, cos_yaw)
# [w,l,h] = exp(w,l,h)   # 恢复真实尺寸
# 输出: boxes_3d, scores_3d, labels_3d, instance_ids
```

---

## 6. 地图头（MapHead）

使用与检测头基本相同的 Sparse4DHead 结构，但针对地图线要素做了适配。

### 6.1 关键差异

| 属性 | 检测头 | 地图头 |
|------|--------|--------|
| num_anchor | 900 | 100 |
| 锚点维度 | 11（3D box） | 40（20 sample × 2D） |
| 锚点初始化 | kmeans_det_900.npy | kmeans_map_100.npy |
| anchor_handler | SparseBox3DKeyPointsGenerator | SparsePoint3DKeyPointsGenerator |
| anchor_encoder | SparseBox3DEncoder(分 dim 编码) | SparsePoint3DEncoder(全连接编码) |
| decouple_attn | True | False |
| num_temp_instances | 600 | 33 |
| refine_yaw | True | N/A |
| reg_weights | [2,2,2, 0.5,0.5,0.5, 0,0, 1,1] | [1.0] × 40 |
| num_dn_groups | 0 | 0 |
| with_instance_id | True | False |
| 回归监督维度 | 11 | 40（20 pts × 2 coords） |

### 6.2 SparsePoint3DKeyPointsGenerator

锚点表示：20 个采样点 × 2D (x,y) → 40 维向量

Keypoints 生成：
- 基础锚点 reshape 为 `[B, 100, 20, 2]`
- 通过可学习 MLP 生成每个采样点的 offset
- 加入 5 个固定高度 `(0, 0.5, -0.5, 1, -1)` 做出 3D keypoints
- 地面高度 `ground_height = -1.84023`（LiDAR 坐标系）

```
key_points = [B, 100, 20×5×3=300, 3]
```

### 6.3 地图目标分配

使用 `HungarianLinesAssigner` + `MapQueriesCost`：
- `cls_cost`: FocalLossCost(weight=1.0)
- `reg_cost`: LinesL1Cost(weight=10.0, beta=0.01, permute=True)

**permute=True** 表示标签是排列不变的（线的前后方向可交换），在匹配时选择最小 cost 的排列。

### 6.4 地图 Loss

```python
loss_cls = FocalLoss(sigmoid, gamma=2.0, alpha=0.25, weight=1.0)
loss_reg = SparseLineLoss:
    loss_line = LinesL1Loss(weight=10.0, beta=0.01)  # smooth L1
```

### 6.5 地图输出解码

```python
SparsePoint3DDecoder.decode()
# 输出: vectors (list of [20,2]), scores, labels
```

---

## 7. 运动预测 + 规划头（MotionPlanningHead）

文件：`motion_planning_head.py`

这是 SparseDrive 的核心创新 —— 并行进行运动预测和规划。

### 7.1 整体结构

```python
class MotionPlanningHead(BaseModule):
    motion_anchor = kmeans_motion_6.npy  # [10,6,12,2]（每个类别有 6 种运动模态）
    plan_anchor   = kmeans_plan_6.npy     # [3,6,6,2]（3 种指令 × 6 种规划模态）
```

### 7.2 输入构建

```
# 1. 从检测头选取 top-50 高置信度的 instance features
det_confidence → topk → instance_feature_selected [B, 50, 256], anchor_embed_selected [B, 50, 256]

# 2. 从地图头选取 top-10 高置信度的 map features
map_confidence → topk → map_instance_feature [B, 10, 256], map_anchor_embed [B, 10, 256]

# 3. 实例队列获取 ego 特征
instance_queue.get() → ego_feature [B, 1, 256], ego_anchor [B, 1, 11], temp_features [B, 51, queue, 256], temp_anchors [B, 51, queue, 11]
```

**总 instance 数量**：`50 (agent) + 1 (ego) = 51`

### 7.3 Motion Anchor 生成

```
motion_anchor = kmeans_motion_6[cls_id]  # 根据检测类别选择对应的运动锚点
motion_anchor = agent2lidar(motion_anchor, bbox)  # 从 Agent 坐标系转到 LiDAR 坐标系
motion_mode_query = MLP(sine_embed(motion_anchor[..., -1, :]))  # [B, 50, 6, 256]
```

### 7.4 Plan Anchor 生成

```
plan_anchor = kmeans_plan_6              # [3, 6, 6, 2] (cmd, mode, ts, xy)
plan_pos = sine_embed(plan_anchor[..., -1, :])
plan_mode_query = MLP(plan_pos).flatten(1,2)  # [B, 1, 3*6=18, 256] → unsqueeze
```

### 7.5 Decoder 操作序列

```python
operation_order = (
    ["temp_gnn", "gnn", "norm", "cross_gnn", "norm", "ffn", "norm"] * 3
    + ["refine"]
)  # 共 3 个 interaction block + 1 个 refine
```

| 操作 | 作用 | Query | Key/Value |
|------|------|-------|-----------|
| `temp_gnn` | 时序交互 | 当前 51 个实例 | 时序队列中缓存的实例 |
| `gnn` | 实例交互 | 全部 51 个实例 | 选出的 top-50 agent 实例 |
| `cross_gnn` | 地图-实例交叉注意力 | 全部 51 个实例 | top-10 地图实例 |
| `norm` | LayerNorm | — | — |
| `ffn` | AsymmetricFFN(256) | — | — |
| `refine` | MotionPlanningRefinementModule | 输出运动和规划 | 详见 7.6 |

**关键设计**：检测和规划共享同一个 instance_feature，通过索引分离：
- `instance_feature[:, :50]`：agent detection features（运动预测用）
- `instance_feature[:, 50:]`：ego feature（规划用）

### 7.6 MotionPlanningRefinementModule

```
motion_query = motion_mode_query + (instance_feature + anchor_embed)[:, :50]
plan_query   = plan_mode_query + (instance_feature + anchor_embed)[:, 50:]

# Motion head
motion_cls = Linear→[1] per mode         → [B, 50, 6]
motion_reg = Linear→[12×2=24] per mode   → [B, 50, 6, 12, 2]  # 位移量

# Planning head  
plan_cls   = Linear→[1] per mode         → [B, 1, 18] (3cmd×6mode)
plan_reg   = Linear→[6×2=12] per mode    → [B, 1, 18, 6, 2]   # 位移量
plan_status = Linear(256→10)             → [B, 1, 10]          # 自车状态
```

**注意**：运动轨迹输出的是**位移量**（velocity），最终轨迹为 `cumsum(displacement)`。

### 7.7 运动预测 Loss

```python
motion_sampler = MotionTarget
  # 使用检测头 Hungarian 匹配的 indices 获取对应 GT
  # 对每个匹配的 anchor，在 6 个 mode 中选择与 GT 最接近的作为正样本 (argmin L2 distance)
  # cls_target = mode_idx (best matching mode)

motion_loss_cls = FocalLoss(sigmoid, gamma=2.0, alpha=0.25, weight=0.2)
  # 分类：预测哪个 mode 最匹配

motion_loss_reg = L1Loss(weight=0.2)
  # 回归：对位移量的 cumsum 与 GT 轨迹的 L1
```

### 7.8 规划 Loss

```python
planning_sampler = PlanningTarget
  # 根据 gt_ego_fut_cmd（one-hot 3 维：直行/左转/右转）选择对应的 cmd branch
  # 在 6 个 mode 中选择与 GT 最接近的作为正样本

plan_loss_cls    = FocalLoss(sigmoid, gamma=2.0, alpha=0.25, weight=0.5)
  # 预测最优 mode
plan_loss_reg    = L1Loss(weight=1.0)
  # 规划轨迹回归
plan_loss_status = L1Loss(weight=1.0)
  # 自车状态回归（速度、加速度等 10 维状态向量）
```

### 7.9 HierarchicalPlanningDecoder（推理）

规划输出在推理时通过**分层选择 + 碰撞感知 rescore**：

```
1. cmd 选择：根据 gt_ego_fut_cmd 选择对应 branch
2. mode 选择：在 6 个 mode 中选择最高分的
3. Collision-aware rescore：
   a. 将 ego 规划轨迹扩展为 3D box（4.08×1.73×1.56）
   b. 将检测到的 agent 运动轨迹也扩展为 3D box
   c. 检查 ego box 与 agent box 是否有碰撞
   d. 碰撞的 mode 分数减去 999（penalty）
   e. 所有 mode 都碰撞时不做惩罚（否则无解）
4. 选择最终无碰撞的规划轨迹
```

---

## 8. InstanceBank —— 检测记忆模块

> ⚠️ **定位澄清**：InstanceBank 主要是**DETR 式可学习先验查询**的存储模块，附带 1 帧时序缓存用于检测稳定性 + 隐式位置继承追踪。它不是"时序建模核心"——真正的多帧时序在 InstanceQueue（第 9 节）。

文件：`instance_bank.py`
所属：DetHead / MapHead **内部最顶层**

### 8.1 存储了什么

| 属性 | Shape | 类型 | 含义 | 是否可学习 |
|------|-------|------|------|-----------|
| `anchor` | `[num_anchor, 11]` (det) / `[num_anchor, 40]` (map) | **nn.Parameter** | k-means 聚类初始化的 900/100 个先验锚点 | ✅ `anchor_grad=True` |
| `instance_feature` | `[num_anchor, 256]` | **nn.Parameter** | 可学习的查询特征（DETR-style） | ✅ `feat_grad=True` |
| `cached_feature` | `[B, num_temp_instances, 256]` | 运行时 buffer | 上帧 top-600/33 实例特征（detach，不反传梯度） | ❌ |
| `cached_anchor` | `[B, num_temp_instances, 11/40]` | 运行时 buffer | 上帧锚点（经坐标变换对齐到当前帧） | ❌ |
| `confidence` | `[B, num_temp_instances]` | 运行时 buffer | 各实例置信度，每帧 ×0.6 衰减 | ❌ |
| `instance_id` | `[B, num_anchor]` | 运行时 buffer | 全局唯一追踪 ID（跨帧关联用） | ❌ |

### 8.2 实现的 4 个功能

1. **DETR 式稀疏检测** —— 900 个可学习 nn.Parameter 作为物体先验 query，通过 6 层 decoder 逐步 refine

2. **时序一致性（Tracking by Attention）** —— `temp_gnn` 中当前帧 900 query 与上帧 top-600 缓存做 cross-attention，让检测在时序上稳定

3. **实例追踪 ID 分配** —— `get_instance_id()` 实现跨帧 ID 继承：
   ```python
   # 上帧 ID 继承到当前帧 top-600 实例
   instance_id[:, :600] = prev_instance_id
   # 新出现的实例分配新 ID (global counter)
   mask = instance_id < 0
   new_ids = arange(num_new) + prev_id
   instance_id[mask] = new_ids
   ```

4. **时序锚点对齐** —— 用 ego 位姿 `T_global_inv @ T_global` 将上帧锚点坐标变换到当前帧坐标系

### 8.3 生命周期

```
每一帧 forward:
  1. get()            — 返回当前锚点 + 缓存时序锚点（拼接为 900+600）
  2. decoder layers   — temp_gnn 用时序锚点做 cross-attention
                         gnn 用当前锚点做 self-attention
                         deformable 从多视图特征采样
                         refine 输出更新后的预测
  3. update()         — 第1个decoder后:
     topk(confidence, 600) → 选取高置信度的与缓存拼接
     用 temporal_mask 判断是否回退到原锚点
  4. cache()          — 保存当前帧 top-600 实例 feature/anchor/confidence
```

### 8.4 时序锚点对齐

```python
T_temp2cur = T_global_inv @ cached_T_global  # 使用 ego 位姿变换
cached_anchor = anchor_projection(cached_anchor, [T_temp2cur])
```

### 8.5 置信度衰减

```python
confidence = max(cached_confidence * 0.6, current_confidence)
# 旧帧置信度逐渐衰减，但新检测可刷新
```

---

## 9. InstanceQueue —— 运动实例队列

文件：`instance_queue.py`
所属：MotionPlanningHead **内部最顶层**

### 9.1 存储了什么

| 属性 | Shape | 类型 | 含义 | 是否可学习 |
|------|-------|------|------|-----------|
| `instance_feature_queue` | `List[B, 50, 256] × queue_length` | 运行时 buffer（List） | DetHead 输出的 top-50 agent 特征（detach）的 4 帧滑窗 | ❌ |
| `anchor_queue` | `List[B, 50, 11] × queue_length` | 运行时 buffer（List） | 对应的检测锚点（detach） | ❌ |
| `ego_feature_queue` | `List[B, 1, 256] × queue_length` | 运行时 buffer（List） | 从 stride-32 特征图池化的 ego 特征 | ❌ |
| `ego_anchor_queue` | `List[B, 1, 11] × queue_length` | 运行时 buffer（List） | Ego 锚点（固定位置尺寸） | ❌ |
| `period` | `[B, 50]` | 运行时 buffer | 每个 agent 被连续追踪的帧数 | ❌ |
| `ego_period` | `[B, 1]` | 运行时 buffer | Ego 已追踪帧数 | ❌ |
| `prev_instance_id` | `[B, 900]` | 运行时 buffer | 上帧 instance_id（用于跨帧匹配） | ❌ |
| `prev_ego_status` | `[B, 10]` | 运行时 buffer | 上帧 ego 状态（速度、加速度等） | ❌ |

### 9.2 Ego 特征提取

```
feature_map = FPN 输出的 stride-32 特征图 [B, 6, 256, 8, 22]
取第一视图 [B, 256, 8, 22]
→ Conv3x3 + BN → Conv3x3 stride2 + BN + ReLU → AvgPool
→ ego_feature [B, 256, 1, 1] → [B, 1, 256]
```

### 9.3 Ego 锚点初始化

```
ego_anchor = [0, 0.5, -1.84+1.56/2, log(4.08), log(1.73), log(1.56), 1, 0, 0, 0, 0]
# 固定位置 (0, 0.5) 高度，车体尺寸 4.08×1.73×1.56
# 上一帧状态中的 VY 会回传
```

### 9.4 时序实例追踪（Motion）

通过 instance_id 匹配跨帧的检测实例：

```python
match = instance_id[..., None] == prev_instance_id[:, None]
# 如果匹配上，保留该实例的缓存特征
# 否则清空（新出现的实例没有时序信息）
```

### 9.5 时序掩码

```python
temp_mask = period < age  # 实际存在的帧才有效
```

---

## 10. InstanceBank vs InstanceQueue：完整对比

| 对比维度 | InstanceBank | InstanceQueue |
|----------|-------------|--------------|
| **所属模块** | DetHead / MapHead | MotionPlanningHead |
| **所在文件** | `models/instance_bank.py` | `models/motion/instance_queue.py` |
| **存的是** | **可学习的 nn.Parameter**（900 个先验锚点 + 查询特征）+ 上帧缓存 buffer | **DetHead 输出的 detach() 特征**的 4 帧 FIFO 滑窗队列 |
| **参数还是 buffer** | `anchor` 和 `instance_feature` 是 **nn.Parameter**（参与反向传播） | 全部是运行时 **buffer**（detach，不反传梯度） |
| **时序深度** | 1 帧——只缓存上帧 top-600/33 | 4 帧——FIFO 队列，保存最近 4 帧的 top-50 agent + 1 ego |
| **存储方式** | 单 tensor 覆盖更新 | List 滑窗，push + pop（超 queue_length 则 pop 最早帧） |
| **核心操作** | `get()` → decoder → `update()` → `cache()` | `prepare_motion/planning()` → decoder → `cache_motion/planning()` |
| **时间对齐方式** | 用 `T_global_inv @ T_global` 做坐标系变换 | 同样用位姿变换，但是对整个队列所有帧做投影 |
| **ID 追踪机制** | `instance_id` 全局计数器，隐式继承 | 显式匹配 `instance_id[..., None] == prev_instance_id[:, None]` |
| **时序作用** | temp_gnn 中**当前帧 900 query** 与**上帧 top-600** 做 cross-attention → 检测稳定 | temp_gnn 中**当前 51 queries** 与**队列中所有帧**做 cross-attention → 运动建模 |
| **生命周期** | 每帧覆盖：当前帧处理完 → 替换缓存 | 持续累积：每帧 push 新数据 → 超长度 pop 旧数据 |
| **典型场景** | 单帧检测 + 单帧跟踪（tracking by attention） | 多帧运动建模 + ego 状态估计 + 碰撞感知规划 |

---

## 11. 追踪机制是如何工作的

> ⚠️ 这是对网络架构最重要的补充问题。InstanceBank 实现的是一种**无显式匹配、基于位置继承的隐式追踪**。和 InstanceQueue 的显式 ID 匹配完全不同。理解这一点对把握 SparseDrive 的时序设计至关重要。

### 核心机制：位置继承 + temp_gnn 一致性

InstanceBank 的追踪**没有使用 Hungarian matching**（那是检测的 loss matching，不是追踪）。它依赖一个简单的假设：**同一物体在相邻帧的置信度排名中是稳定的**。

**完整链条**（追踪代码在 `instance_bank.py:148-259`）：

```
══════════════════════════════════════════════════════════════
  帧 N 结束时
══════════════════════════════════════════════════════════════

1. 经过 6 层 decoder → instance_feature [B, 900, 256], confidence [B, 900]

2. cache() (行 195-221):
   confidence = max(prev_conf * 0.6, current_conf)
   topk(confidence, 600) → cached_feature [B, 600, 256]
                           cached_anchor  [B, 600, 11]

3. get_instance_id() (行 223-241):
   instance_id = full([B, 900], -1)
   instance_id[:, :600] = self.instance_id    ← 位置[0:599]继承上帧 ID
   new_ids = arange(num_new) + prev_id        ← 位置[600:899]分配新 ID
   self.prev_id += num_new

4. update_instance_id() (行 243-259):
   topk(confidence, 600, instance_id) → self.instance_id [B, 600]

══════════════════════════════════════════════════════════════
  帧 N+1 开始时
══════════════════════════════════════════════════════════════

5. get() (行 84-146):
   返回 cached_feature [B, 600, 256] 和 cached_anchor [B, 600, 11]
   (anchor 已通过 anchor_projection 对齐到当前帧坐标系)

6. temp_gnn (detection3d_head.py:265-276):
   query=当前900 instances → key=value=上帧600 cached instances
   这是 cross-attention，模型学习跨帧特征关联

7. update() (行 148-193):
   topk(confidence, 300) → 当前帧 top-300
   merged = cat([cached_600, 当前top300], dim=1) → 900
   instance_feature = where(temporal_mask, merged, original)

8. 后续 5 层 decoder refine

9. get_instance_id():
   instance_id[:, :600] = self.instance_id ← 再次位置继承
   → 位置[0:599]继承了帧 N 的 ID
```

### 追踪为什么能工作？

| 保障机制 | 作用 |
|----------|------|
| `update()` 的 **concat 顺序固定** | `[cached_600, current_top300]` — 位置 [0:599] 始终是上帧实例 |
| `get_instance_id()` 的 **位置继承** | 前 600 个位置自动获得上帧 ID |
| **temp_gnn cross-attention** | 当前 900 query 看上帧 600，让模型学习保持特征一致性 |
| **confidence_decay=0.6** | 稳定物体保持高排名，已消失的物体被淘汰 |

### 追踪的风险

**没有显式匹配！** 如果同一物体在帧间排名剧烈跳动（例如从第 23 名掉到第 612 名），它的 ID 就丢失了。追踪质量完全取决于：

1. **检测稳定性**：模型能否在相邻帧对同一物体给出相似的置信度排名
2. **temp_gnn 效果**：cross-attention 是否能让当前帧特征 "回忆" 上帧特征

**对比 InstanceQueue 的显式匹配**：

```python
# InstanceQueue: 显式 ID 匹配
match = instance_id[..., None] == prev_instance_id[:, None]
# 匹配上了 → 保留时序特征；没匹配上 → 清空

# InstanceBank: 位置继承
instance_id[:, :600] = self.instance_id  # 无条件继承
# 没有验证！只靠置信度排名的稳定性
```

### 没有 GT 监督追踪

Instance ID 是**纯算法分配**的（`prev_id += num_new` 全局递增计数器），没有 tracking loss，没有 tracking GT。模型通过检测 loss 间接学习时序一致性——**稳定的检测意味着稳定的排名意味着连续的 ID**。这是一个"免费"的追踪，但准确度波动是它的代价。

---

## 12. 损失函数总表

### 12.1 Stage1 训练（Det + Map，无 Motion）

| Loss | 组件 | 权重 | 公式要点 |
|------|------|------|---------|
| `det_loss_cls_N` | FocalLoss | 2.0 | γ=2.0, α=0.25 |
| `det_loss_box_N` | L1Loss | 0.25 | 在 encoded box 上 |
| `det_loss_cns_N` | CrossEntropyLoss(sigmoid) | — | centerness |
| `det_loss_yns_N` | GaussianFocalLoss | — | yawness |
| `map_loss_cls_N` | FocalLoss | 1.0 | γ=2.0, α=0.25 |
| `map_loss_line_N` | LinesL1Loss(smooth L1) | 10.0 | β=0.01 |
| `loss_dense_depth` | L1 (sum scale) | 0.2 | 仅 foreground |

### 12.2 Stage2 训练（Det + Map + Motion + Plan）

| Loss | 组件 | 权重 | 公式要点 |
|------|------|------|---------|
| 上述全部 + | | | |
| `motion_loss_cls_N` | FocalLoss | 0.2 | 预测最优运动模态 |
| `motion_loss_reg_N` | L1Loss | 0.2 | cumsum 轨迹与 GT |
| `planning_loss_cls_N` | FocalLoss | 0.5 | 预测最优规划模态 |
| `planning_loss_reg_N` | L1Loss | 1.0 | 规划轨迹 |
| `planning_loss_status_N` | L1Loss | 1.0 | 10 维 ego 状态 |

### 12.3 监督信号与 GT 对应

| GT Key | Shape | 来源 | 用于 |
|--------|-------|------|------|
| `img` | `[B,6,3,704,256]` | 多视图图像 | 模型输入 |
| `gt_depth` | `[B×6, 1, H, W]` × 3 | LiDAR 投影 | 深度辅助监督 |
| `gt_bboxes_3d` | List of `[N_i, 11]` | nuScenes | 3D 检测 |
| `gt_labels_3d` | List of `[N_i]` | nuScenes | 检测分类 |
| `gt_map_labels` | List of `[N_i_map]` | nuScenes map | 地图分类 |
| `gt_map_pts` | List of `[N_i_map, 20, 2]` | vectorize | 地图回归 |
| `gt_agent_fut_trajs` | `[B, N, 12, 2]` | nuScenes | 运动预测 |
| `gt_agent_fut_masks` | `[B, N, 12]` | 存在掩码 | 运动预测（填充 mask） |
| `gt_ego_fut_trajs` | `[B, 6, 2]` | nuScenes | 规划 |
| `gt_ego_fut_masks` | `[B, 6]` | 存在掩码 | 规划 |
| `gt_ego_fut_cmd` | `[B, 3]` one-hot | 命令 | 规划（直行/左转/右转） |
| `ego_status` | `[B, 10]` | 自车状态 | 规划状态回归 |
| `projection_mat` | `[B, 6, 4, 4]` | 相机参数 | deformable attention |
| `image_wh` | `[B, 6, 2]` | 图像尺寸 | 投影归一化 |
| `T_global/T_global_inv` | `[B, 4, 4]` | ego 位姿 | 时序对齐 |
| `timestamp` | `[B]` | 时间戳 | 时间差计算 |
| `focal` | `[B, 6]` | 相机焦距 | 深度分支 |

---

## 13. 文本网络结构图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  INPUT: 6× 多视图图像 [B, 6, 3, 256, 704] + LiDAR点云 (仅训练)               │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ResNet50 + FPN                                                              │
│  输出 4 层多尺度特征图 [B, 6, 256, H_i, W_i]  stride ∈ {4,8,16,32}         │
│  ⚠️ 这不是24张独立图 — 6个相机作为统一dim，deformable attn跨全部6cam×4level   │
│  在一个操作内完成，权重 [B,900,6,4,7,8] 自动决定各anchor关注哪些view/level    │
└──────┬───────────────────────────────────────────────┬───────────────────────┘
       │                                               │
       ▼                                               ▼
┌──────────────┐                             ┌──────────────────────────────────┐
│ DenseDepthNet│                             │      SparseDriveHead             │
│ 3-scale depth│                             │                                  │
│  └→ depth loss│                            │  ┌────────────────────────────┐  │
└──────────────┘                             │  │  DetHead (Sparse4DHead)     │  │
                                              │  │                            │  │
                                              │  │  InstanceBank (存的是:      │  │
                                              │  │  ├── anchor [900,11]        │  │
                                              │  │  │    → nn.Parameter(可学习)│  │
                                              │  │  ├── instance_feat [900,256]│  │
                                              │  │  │    → nn.Parameter(可学习)│  │
                                              │  │  ├── cached_feat [B,600,256]│  │
                                              │  │  │    → 上帧buffer(detach)  │  │
                                              │  │  └── instance_id [B,900]    │  │
                                              │  │       → 全局追踪ID         │  │
                                              │  └────────┬───────────────────┘  │
                                              │           │ get()返回900+600     │
                                              │           ▼                     │
                                              │  AnchorEncoder → [B,900,256]    │
                                              │           │                     │
                                              │           ▼                     │
                                              │  6× Decoder Layers              │
                                              │  ┌────────────────────────┐     │
                                              │  │ temp_gnn ←→ 上帧600   │     │
                                              │  │ gnn ←→ 当前帧900       │     │
                                              │  │ deformable: 跨6cam×4lev│     │
                                              │  │    权重 [900,6,4,7,8]  │     │
                                              │  │ ffn → refine output    │     │
                                              │  └────────────────────────┘     │
                                              │           │                     │
                                              │           ▼                     │
                                              │  Output: anchors + cls + feat   │
                                              │  + instance_id + quality        │
                                              └────────────────────────────────┘
                                              │
                                              │  ┌────────────────────────────┐  │
                                              │  │  MapHead (Sparse4DHead)    │  │
                                              │  │  InstanceBank (存的是:      │  │
                                              │  │  ├── anchor [100,40]       │  │
                                              │  │  │    → nn.Parameter(可学习)│  │
                                              │  │  └── cached [B,33,...]     │  │
                                              │  │  6× Decoder Layers (同类)   │  │
                                              │  │  → line pts [100,40] + cls │  │
                                              │  └────────────────────────────┘  │
                                              │                                  │
                                              │  ┌────────────────────────────┐  │
                                              │  │  MotionPlanningHead        │  │
                                              │  │                            │  │
                                              │  │  top-50 det feats          │  │
                                              │  │  + top-10 map feats        │  │
                                              │  │  + 1 ego feat (共51)       │  │
                                              │  │         │                  │  │
                                              │  │         ▼                  │  │
                                              │  │  InstanceQueue (存的是:     │  │
                                              │  │  ├── feat_queue[4][B,50,256]│  │
                                              │  │  │    → detach buffer FIFO  │  │
                                              │  │  ├── ego_queue[4][B,1,256] │  │
                                              │  │  │    → 特征图池化 buffer   │  │
                                              │  │  └── period[B,50]+ego_period│  │
                                              │  │       → 全部是运行Buffer    │  │
                                              │  └────────┬──────────────────┘  │
                                              │           │ get()返回队列4帧     │
                                              │           ▼                     │
                                              │  3× Interaction Blocks           │
                                              │  ┌────────────────────────┐     │
                                              │  │ temp_gnn ←→ 队列4帧   │     │
                                              │  │ gnn: 51 queries间交互  │     │
                                              │  │ cross_gnn ←→ 10 map   │     │
                                              │  │ ffn → MotionPlanRefine│     │
                                              │  └────────────────────────┘     │
                                              │           │                     │
                                              │           ▼                     │
                                              │  MotionPlanningRefine            │
                                              │  ├── motion_cls  [B,50,6]       │
                                              │  ├── motion_reg  [B,50,6,12,2]  │
                                              │  ├── plan_cls    [B,1,18]        │
                                              │  ├── plan_reg    [B,1,18,6,2]    │
                                              │  └── plan_status [B,1,10]        │
                                              │                                  │
                                              │  ┌── det: boxes + labels         │
                                              │  ├── map: vectors + labels       │
                                              │  ├── motion: trajs + scores      │
                                              │  └── plan: trajectory + status   │
                                              └──────────────────────────────────┘

                              === 损失计算 ===

  detection loss     map loss        depth loss      motion loss      plan loss
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │cls(Focal)│    │cls(Focal)│    │L1(depth) │    │cls(Focal)│    │cls(Focal)│
  │box(L1)   │    │line(L1)  │    │fg only   │    │reg(L1)   │    │reg(L1)   │
  │cns(CE)   │    │          │    │          │    │          │    │status(L1)│
  │yns(GF)   │    │          │    │          │    │          │    │          │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘

                              === 训练策略 ===
  Stage1: det + map                     (100 epochs)
  Stage2: det + map + motion + plan     (10 epochs, 从 Stage1 加载)
```

---

## 附录 A：关键模块文件映射

| 模块 | 文件路径 |
|------|---------|
| 主模型 (SparseDrive) | `projects/mmdet3d_plugin/models/sparsedrive.py` |
| Head 总调度 (SparseDriveHead) | `projects/mmdet3d_plugin/models/sparsedrive_head.py` |
| 检测头 (Sparse4DHead) | `projects/mmdet3d_plugin/models/detection3d/detection3d_head.py` |
| 检测 decoder (SparseBox3DDecoder) | `projects/mmdet3d_plugin/models/detection3d/decoder.py` |
| 检测 target (SparseBox3DTarget) | `projects/mmdet3d_plugin/models/detection3d/target.py` |
| 检测模块 (refine/encoder/keypoints) | `projects/mmdet3d_plugin/models/detection3d/detection3d_blocks.py` |
| 检测 loss (SparseBox3DLoss) | `projects/mmdet3d_plugin/models/detection3d/losses.py` |
| 地图 decoder (SparsePoint3DDecoder) | `projects/mmdet3d_plugin/models/map/decoder.py` |
| 地图 target (SparsePoint3DTarget) | `projects/mmdet3d_plugin/models/map/target.py` |
| 地图模块 (refine/encoder/keypoints) | `projects/mmdet3d_plugin/models/map/map_blocks.py` |
| 地图 match cost | `projects/mmdet3d_plugin/models/map/match_cost.py` |
| 地图 loss | `projects/mmdet3d_plugin/models/map/loss.py` |
| 运动/规划 head (MotionPlanningHead) | `projects/mmdet3d_plugin/models/motion/motion_planning_head.py` |
| 运动/规划 refine | `projects/mmdet3d_plugin/models/motion/motion_blocks.py` |
| 运动/规划 decoder | `projects/mmdet3d_plugin/models/motion/decoder.py` |
| 运动/规划 target | `projects/mmdet3d_plugin/models/motion/target.py` |
| InstanceQueue | `projects/mmdet3d_plugin/models/motion/instance_queue.py` |
| InstanceBank | `projects/mmdet3d_plugin/models/instance_bank.py` |
| DeformableFeatureAggregation | `projects/mmdet3d_plugin/models/blocks.py` |
| DenseDepthNet | `projects/mmdet3d_plugin/models/blocks.py` |
| AsymmetricFFN | `projects/mmdet3d_plugin/models/blocks.py` |
| FlashAttention | `projects/mmdet3d_plugin/models/attention.py` |
| Box3D 常量定义 | `projects/mmdet3d_plugin/core/box3d.py` |
| 数据 pipeline | `projects/mmdet3d_plugin/datasets/pipelines/` |
| 评估 | `projects/mmdet3d_plugin/datasets/evaluation/` |

## 附录 B：Box3D 编码格式

```python
# 未解码的 anchor/box (11 维):
X, Y, Z,          # 位置 (0-2)
W, L, H,          # 尺寸 (3-5)
SIN_YAW, COS_YAW, # 朝向 (6-7)
VX, VY, VZ        # 速度 (8-10)

# 解码后 (YAW 取代 SIN_YAW+COS_YAW):
X, Y, Z,          # 位置
W, L, H,          # 尺寸 (exp 后为真实值)
YAW,              # 朝向角 (atan2)
VX, VY, VZ        # 速度
```

## 附录 C：注意力机制说明

**decouple_attn**：当 `decouple_attn=True` 时，在 MultiheadFlashAttention 中：
- query 和 key 拼接位置编码：`Q' = [Q; Q_pos]`, `K' = [K; K_pos]`
- value 通过 fc_before(256→512) 升维，attention 后在 512 维度上计算，再通过 fc_after(512→256) 降维
- 效果：解耦 content 和 position 的注意力计算

**DeformableFeatureAggregation**：基于 3D 锚点生成 3D keypoints，投影到多视图多尺度 2D 特征图采样，通过可学习的 sparse attention weights 融合多视图多尺度特征。
