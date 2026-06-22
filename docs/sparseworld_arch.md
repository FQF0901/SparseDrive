# SparseWorld 架构伪代码

> 基于 SparseDrive 代码仓库 (ec0225d) + SparseDrive ICRA 2025 Paper + SparseWorld arxiv:2605.24394 Paper
> 所有 tensor shape 均从代码追溯确认；标记 `(paper)` 的为从 paper 推断
> 核心代码文件: `motion_planning_head.py`(483行), `instance_queue.py`(213行), `sparsedrive_head.py`(124行)

---

## 1. 总体结构

```
SparseWorld = SparseDrive + 3 个新模块 + 2 个修改模块

复用模块 (完全不变):
  ResNet50 Backbone, FPN Neck, DenseDepthNet
  Sparse4DHead (det) — InstanceBank 900 anchors + 600 temporal
  Sparse4DHead (map) — InstanceBank 100 anchors + 33 temporal
  DeformableFeatureAggregation, MultiheadFlashAttention, AsymmetricFFN

修改模块:
  InstanceQueue — 原来只存历史4帧, 现扩展为存历史+未来实例
  MotionPlanningHead — 增加 world_model_forward() + refinement 分支

新增模块 (3个文件):
  sparse_dreamer.py — SparseDreamer (GIA + SWD)
  safety_loss.py — Safety-Critical Loss (SCL)
  adaptive_trajectory_selection.py — Adaptive Trajectory Selection (ATS)
```

---

## 2. Stream-Based 数据管线

### 2.1 SparseDrive 现有时间管线 (基线)

```python
# ============================================================
# SparseDrive stream pipeline (每帧)
# 文件: sparsedrive_head.py:41-69
# ============================================================

def forward(frame_t):
    # step 1: 图像编码
    feature_maps = backbone(frame_t.images)  # [B,6,256,H,W]×4scales
    feature_maps = neck(feature_maps)

    # step 2: 稀疏感知
    det_output = det_head(feature_maps, metas)
    # det_output = {
    #   "instance_feature": [B,900,256],
    #   "anchor_embed":     [B,900,256],
    #   "classification":   list([B,900,10]) × 6,
    #   "prediction":       list([B,900,11]) × 6,
    #   "quality":          list([B,900,2]|None) × 6,
    #   "instance_id":      [B,900],
    # }

    map_output = map_head(feature_maps, metas)
    # map_output = {
    #   "instance_feature": [B,100,256],
    #   "anchor_embed":     [B,100,256],
    #   "classification":   list([B,100,3]) × 6,
    #   "prediction":       list([B,100,40]) × 6,
    # }

    # step 3: motion + planning (使用 InstanceQueue)
    motion_output, planning_output = motion_plan_head(
        det_output, map_output, feature_maps, metas,
        det_head.anchor_encoder,
        det_head.instance_bank.mask,
        det_head.instance_bank.anchor_handler,
    )
    return det_output, map_output, motion_output, planning_output
```

### 2.2 SparseWorld 扩展后的时间管线

```python
# ============================================================
# SparseWorld stream pipeline (每帧 + 未来预测)
# ============================================================

def forward_sparseworld(frame_t, is_training=True):
    # ── 阶段 1: Baseline 前向 ──
    det_output, map_output, motion_output, planning_output = forward(frame_t)
    I_0 = {
        "agent": {
            "anchor":  det_output["prediction"][-1],       # [B,900,11]
            "feature": det_output["instance_feature"],       # [B,900,256]
        },
        "map": {
            "anchor":  map_output["prediction"][-1],        # [B,100,40]
            "feature": map_output["instance_feature"],       # [B,100,256]
        },
        "ego": {
            "anchor":  ego_anchor,                           # [B,1,11]
            "feature": ego_feature,                          # [B,1,256]
        },
    }
    C_0 = {
        "velocity":   ego_status[:,6],    # [B] scalar (paper: 来自ego status)
        "trajectory": planning_output["prediction"][-1],  # [B,18,6,6,2] → 取 selected
        "command":    cmd,                 # [B] (turn_left/straight/right)
    }
    tau_base = planning_output["final_planning"]  # SparseDrive baseline trajectory

    # ── 阶段 2: 自回归未来预测 ──
    I_future = []  # 存储 {I_1, I_2, ..., I_f}
    I_current = I_0
    C_current = C_0
    for k in range(f):  # f=4 frames (paper)
        if is_training:
            # teacher forcing: 用 GT 未来 action conditions
            C_k = get_gt_action_conditions(frame_{t + k + 1})
        else:
            # 推理: FMP 从预测实例推断 action conditions
            C_k = future_motion_planner(I_current)

        # SparseDreamer 预测下一帧实例
        I_next = sparse_dreamer(
            historical_instances = [I_{0-m}..I_current],  # m=4 from IMQ (paper: m待确认)
            action_conditions = C_k,
            ego_status = ego_status,
        )
        I_future.append(I_next)
        I_current = I_next

    # ── 阶段 3: Motion Planning Refinement ──
    tau_refined_agents = refine_motion(
        current_instances = I_0,
        future_instances = I_future,
    )
    tau_refined_ego = refine_planning(
        ego_feature = I_0.ego.feature,
        future_features = [I.agent.feature for I in I_future],
    )
    tau_future = FMP_last_planning  # FMP 最后一步的 planning 输出

    # ── 阶段 4: Adaptive Trajectory Selection ──
    tau_final = ats_select(
        candidates = [tau_base, tau_future, tau_refined_ego],
        agent_predictions = tau_refined_agents,
        future_agent_boxes = [I.agent.anchor for I in I_future],
    )
    return tau_final, I_future, tau_refined_agents
```

---

## 3. SparseDreamer 模块

```python
# ============================================================
# 文件: models/motion/sparse_dreamer.py
# 类: SparseDreamer
# 注册: @PLUGIN_LAYERS or standalone nn.Module
# ============================================================

class SparseDreamer(nn.Module):
    """
    Sparse Dreamer: 在 latent space 中自回归预测未来实例 (agent + map)
    论文 Fig.3 详述架构
    """
    def __init__(self, embed_dims=256, num_decoder=6, num_heads=8):
        # ── GIA: Global Instance Alignment ──
        self.gia = GlobalInstanceAlignment()

        # ── SWD: SparseWorld Decoder ──
        # N 个 transformer decoder layers (paper: N未明确, 推测=6)
        # 每层: self-attn → temp-cross-attn → action-cross-attn
        #       → FFN → classification + refinement
        self.layers = nn.ModuleList([
            SparseWorldDecoderLayer(embed_dims, num_heads)
            for _ in range(num_decoder)
        ])

        # ── Temporal embeddings (paper: Eq3-4 相关) ──
        self.temp_embed = nn.Embedding(queue_len, embed_dims)  # E_time ∈ R^{m×256}
        self.pos_embed = nn.Linear(2, embed_dims)  # E_pos: (Δx,Δy) → 256

        # ── Action embedding (paper: Fourier embeddings from Drive-OccWorld) ──
        self.action_embed = FourierActionEmbedding(embed_dims)

        # ── Anchor encoder (复用 detection 的 SparseBox3DEncoder) ──
        self.anchor_encoder = SparseBox3DEncoder(vel_dims=3, embed_dims=256)

        # ── Refinement modules (复用现存模块) ──
        self.agent_refine = SparseBox3DRefinementModule(embed_dims, num_cls=10)
        self.map_refine = SparsePoint3DRefinementModule(embed_dims, num_sample=20, num_cls=3)

        # ── FFN (复用 AsymmetricFFN) ──
        self.ffn = AsymmetricFFN(256*2, embed_dims=256, feedforward_channels=1024)

    def forward(self,
        historical_instances: List[Dict],  # [{agent, map, ego}] × (m+1)
        action_conditions: Dict,            # {velocity, trajectory, command}
    ) -> Dict:
        """
        Input:
          historical_instances: 长度 m+1 的列表, 每项:
            agent: {anchor:[B,N_agent,11], feature:[B,N_agent,256]}
            map:   {anchor:[B,N_map,40], feature:[B,N_map,256]}
            ego:   {anchor:[B,1,11], feature:[B,1,256]}
          action_conditions:
            velocity:   [B]           ego speed scalar
            trajectory: [B, T, 2]     planned trajectory
            command:    [B]           0=left, 1=straight, 2=right

        Output:
          I_next: {agent, map, ego} — 未来下一帧的实例
        """
        # ── step 1: GIA — 将所有历史实例投影到"当前帧"坐标系 ──
        # (paper Eq.2-4): 用 ego-motion + velocity 做 anchor 预投影
        projected_anchors = []
        for t, I_t in enumerate(historical_instances):
            A_proj = self.gia(
                I_t.agent.anchor,     # [B,N_agent,11]
                I_t.ego.anchor,       # [B,1,11] (用于 ego-motion)
                action_conditions,      # 用于 velocity compensation
                dt = t * 0.5,          # 时间间隔 (nuScenes 2Hz)
            )
            projected_anchors.append(A_proj)

        # ── step 2: 构建初始 query (当前帧实例) ──
        I_0 = historical_instances[-1]
        query_agent_feat = I_0.agent.feature  # [B,N_agent,256]
        query_map_feat = I_0.map.feature      # [B,N_map,256]
        query_ego_feat = I_0.ego.feature      # [B,1,256]
        # concatenate all instances: [B, N_agent+N_map+1, 256]
        query_feat = torch.cat([query_agent_feat, query_map_feat, query_ego_feat], dim=1)

        # ── step 3: 构建历史 key/value (用于 temporal-cross-attn) ──
        # 取前 m 帧的历史实例特征 (paper: m 从 IMQ 中选择)
        hist_feat = torch.stack([I.agent.feature for I in historical_instances[:-1]], dim=2)
        # hist_feat: [B, N_agent, m, 256]

        # ── step 4: 编码 action conditions + temporal embeddings ──
        action_emb = self.action_embed(
            action_conditions.velocity,
            action_conditions.trajectory,
            action_conditions.command,
        )  # [B, 256]

        # temporal embedding: E_time (learnable) + E_pos (relative displacement)
        time_idx = torch.arange(m, device=query_feat.device)
        E_time = self.temp_embed(time_idx)  # [m, 256]
        E_pos = self.pos_embed(displacements)  # [B, N_agent, m, 256]

        # ── step 5: SparseWorld Decoder (N layers) ──
        for layer_idx, layer in enumerate(self.layers):
            # anchor embedding for current query
            anchor_query = torch.cat([
                I_0.agent.anchor, I_0.map.anchor, I_0.ego.anchor
            ], dim=1)  # [B, N_agent+N_map+1, 11|40]
            anchor_embed = self.anchor_encoder(anchor_query)  # [B, N, 256]

            # 5a: self-attention (当前实例之间)
            query_feat = layer.self_attn(
                query = query_feat,           # [B, N, 256]
                key   = query_feat,
                value = query_feat,
                query_pos = anchor_embed,
            )

            # 5b: temporal-cross-attention (当前 → 历史)
            query_feat = layer.temp_cross_attn(
                query = query_feat[:, :N_agent],   # [B, N_agent, 256]
                key   = hist_feat + E_time + E_pos, # [B, N_agent, m, 256]
                value = hist_feat,
                query_pos = anchor_embed[:, :N_agent],
            )

            # 5c: action-conditioned cross-attention
            # (paper: Fourier embedding injected into attention)
            query_feat = layer.action_cross_attn(
                query = query_feat,
                key   = action_emb[:, None, :],  # [B, 1, 256]
                value = action_emb[:, None, :],
            )

            # 5d: FFN
            query_feat = self.ffn(query_feat)

            # 5e: Refinement (classification + box/polyline decode)
            # 仅最后一层做? 或每层都做?(paper未明确, 参考Sparse4DHead每层都做)
            agent_cls, agent_box = self.agent_refine(
                query_feat[:, :N_agent] + anchor_embed[:, :N_agent]
            )  # cls:[B,N_agent,10], box:[B,N_agent,11]
            map_cls, map_pts = self.map_refine(
                query_feat[:, N_agent:N_agent+N_map] + anchor_embed[:, N_agent:N_agent+N_map]
            )  # cls:[B,N_map,3], pts:[B,N_map,40]

        # ── step 6: 组装输出 ──
        I_next = {
            "agent": {"anchor": agent_box, "feature": query_feat[:, :N_agent]},
            "map":   {"anchor": map_pts,   "feature": query_feat[:, N_agent:N_agent+N_map]},
            "ego":   {"anchor": ego_box,   "feature": query_feat[:, -1:]},
        }
        return I_next


class GlobalInstanceAlignment(nn.Module):
    """
    GIA: 在 SparseDreamer 预测未来帧之前, 将历史 anchor 预投影
    利用 ego-motion compensation + agent velocity adjustment
    paper Eq.(2)-(4)
    """
    def forward(self, anchor, ego_anchor, action_conditions, dt):
        """
        Input:
          anchor: [B, N_agent, 11] — agent anchors
          ego_anchor: [B, 1, 11] — ego vehicle anchor
          action_conditions: Dict
          dt: float — 时间间隔 (0.5s for nuScenes)

        Output:
          projected_anchor: [B, N_agent, 11]
        """
        # agent 自身位移: ce_comp = ce - V * Δt   (paper Eq.2)
        V = anchor[..., VX:VZ+1]          # [B,N_agent,3]
        center = anchor[..., X:Z+1]        # [B,N_agent,3]
        ce_comp = center - V * dt          # [B,N_agent,3]

        # ego-motion 旋转 + planner 位移 (paper Eq.3)
        omega = ego_anchor[..., SIN_YAW:COS_YAW+1]  # [B,1,2] → yaw angle
        d_xy = action_conditions["trajectory"][:, 0, :]  # [B,2] 使用 planner 位移
        Rz = rotation_matrix_z(omega)  # [B, 2, 2]
        ce_prime = Rz @ ce_comp[..., :2].unsqueeze(-1) + d_xy[..., None]  # [B,N_agent,2]

        # yaw 旋转 (paper Eq.4)
        yaw = anchor[..., SIN_YAW:COS_YAW+1]  # [B,N_agent,2]
        yaw_prime = Rz @ yaw.unsqueeze(-1)    # [B,N_agent,2]

        # velocity 旋转 (同理)
        vel = V[..., :2]  # [B,N_agent,2] (仅旋转水平速度)
        vel_prime = Rz @ vel.unsqueeze(-1)

        # 静态 map elements: V=0, 仅做 ego-motion compensation
        # map anchors [B,N_map,40] — 20 points × 2 coords, reshape 后同样处理
        projected = torch.cat([
            ce_prime.squeeze(-1),        # X, Y
            center[..., 2:3],            # Z (不变)
            anchor[..., W:H+1],          # W, L, H (不变)
            yaw_prime.squeeze(-1),       # SIN_YAW, COS_YAW
            vel_prime.squeeze(-1),       # VX, VY
            V[..., 2:3],                 # VZ (不变)
        ], dim=-1)
        return projected


class FourierActionEmbedding(nn.Module):
    """
    Action conditions 编码: 参考 Drive-OccWorld 的 Fourier embedding
    Input: (velocity[B], trajectory[B,T,2], command[B])
    Output: [B, embed_dims]
    """
    def __init__(self, embed_dims=256):
        # velocity: scalar → Fourier embed → Linear → 256
        # trajectory: [T,2] → flatten → Fourier embed → Linear → 256
        # command: 3-class embed → Linear → 256
        # concatenate → Linear → embed_dims
        ...


class SparseWorldDecoderLayer(nn.Module):
    """
    SWD 的单个 decoder layer
    结构: self-attn → temp-cross-attn → action-cross-attn → FFN
    self-attn 和 temp-cross-attn 复用 MultiheadFlashAttention
    action-cross-attn 在 attention 中注入 Fourier-encoded action conditions
    """
    def __init__(self, embed_dims, num_heads):
        self.self_attn = MultiheadFlashAttention(embed_dims, num_heads)
        self.temp_cross_attn = MultiheadFlashAttention(embed_dims, num_heads)
        self.action_cross_attn = ActionConditionedAttention(embed_dims, num_heads)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
```

---

## 4. Future Motion Planner (FMP)

```python
# ============================================================
# FMP: 与 baseline MotionPlanner 架构完全相同
# 从未来实例推断 action conditions: (velocity, trajectory, steering)
# ============================================================

class FutureMotionPlanner(nn.Module):
    """
    FMP 复用 baseline motion planner 的架构 (3层: temp_gnn→gnn→cross_gnn→ffn)
    Input: 未来实例 I_{t+k} (来自 SparseDreamer 输出)
    Output: action conditions C_{t+k+1} = (v, τ, s)
    """
    def __init__(self, embed_dims=256):
        # 复用 motion_planning_head 的解码器架构
        # but WITHOUT InstanceQueue (直接输入未来实例)
        self.motion_layers = nn.ModuleList([...])  # 3 decoder layers
        self.refine = MotionPlanningRefinementModule(
            embed_dims, fut_ts=12, fut_mode=6,
            ego_fut_ts=6, ego_fut_mode=6,
        )
        self.ego_status_decoder = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10),  # ego status 10-dim
        )

    def forward(self, I_k: Dict) -> Dict:
        """
        Input:
          I_k: {agent, map, ego} instances from SparseDreamer
             agent: {anchor:[B,N,11], feature:[B,N,256]}
             ego:   {anchor:[B,1,11], feature:[B,1,256]}

        Output:
          C_k: {velocity, trajectory, command} action conditions
              velocity:   [B] scalar
              trajectory: [B, T, 2] (12 timesteps)
              command:    [B] (0=left, 1=straight, 2=right)
        """
        # ── 构建 agent+ego 联合 query ──
        agent_feat = I_k["agent"]["feature"]    # [B,N_agent,256]
        ego_feat = I_k["ego"]["feature"]        # [B,1,256]
        instance_feat = torch.cat([agent_feat, ego_feat], dim=1)  # [B,N_agent+1,256]

        agent_anchor = I_k["agent"]["anchor"]
        ego_anchor = I_k["ego"]["anchor"]
        anchor = torch.cat([agent_anchor, ego_anchor], dim=1)  # [B,N_agent+1,11]
        anchor_embed = self.anchor_encoder(anchor)

        # ── 通过 3 层 decoder (复用 motion planning head 的 attention pattern) ──
        for layer in self.motion_layers:
            instance_feat = layer(instance_feat, anchor_embed)

        # ── Refine: 输出 motion/planning predictions ──
        motion_query = self.motion_mode_query + instance_feat[:, :-1]
        plan_query = self.plan_mode_query + instance_feat[:, -1:]
        motion_cls, motion_reg, plan_cls, plan_reg, ego_status = self.refine(
            motion_query, plan_query, instance_feat[:, -1:], anchor_embed[:, -1:]
        )

        # ── 从 ego_status 和 plan_reg 提取 action conditions ──
        velocity = ego_status[..., 6]           # [B]
        trajectory = plan_reg[argmax(plan_cls)] # [B, T, 2]
        command = plan_cls.flatten(1,2).argmax(-1) // 6  # [B] 0/1/2

        return {"velocity": velocity, "trajectory": trajectory, "command": command}
```

---

## 5. Motion Planning Refinement

```python
# ============================================================
# 修改: models/motion/motion_planning_head.py
# 新增方法: refine_motion(), refine_planning()
# ============================================================

# ── 修改 InstanceQueue.get() ──
# 原来: 仅返回历史实例 (temp_instance_feature, temp_anchor)
# 新增: 返回未来实例 (future_instance_feature, future_anchor)
# 实现: 在原有 queue 基础上新增 self.future_feature_queue, self.future_anchor_queue
# 在 prepare_motion() 之后, call push_future() 将 SparseDreamer 输出入队

class InstanceQueue(nn.Module):
    # ... 原有代码 ...

    def push_future(self, I_future_list):
        """
        存入未来预测实例
        I_future_list: [{agent, map, ego}] × f frames
        """
        for I_k in I_future_list:
            self.future_feature_queue.append(I_k["agent"]["feature"].detach())
            self.future_anchor_queue.append(I_k["agent"]["anchor"].detach())

    def get_future(self):
        """
        获取所有未来帧实例 (用于 refinement)
        Returns:
          future_features: [B, N_agent, f, 256]
          future_anchors:  [B, N_agent, f, 11]
        """
        return (
            torch.stack(self.future_feature_queue, dim=2),
            torch.stack(self.future_anchor_queue, dim=2),
        )


# ── MotionPlanningHead 新增方法 ──

def refine_motion(self, I_0, I_future):
    """
    Motion Prediction Refinement
    用未来实例 {I_1..I_f} 替代历史实例, 重新运行 motion predictor
    paper: Sec III-C "Motion Prediction Refinement"

    Input:
      I_0: 当前帧实例 {agent, map, ego}
      I_future: 未来帧实例列表 [{agent, map, ego}] × f

    Output:
      tau_refined_agents: [B, N_agent, K_m, T_m, 2] 精炼后的 agent 轨迹
    """
    # 构建未来 temporal instances (替代历史的 temp_instance_feature)
    future_feats = torch.stack([I["agent"]["feature"] for I in I_future], dim=2)
    future_anchors = torch.stack([I["agent"]["anchor"] for I in I_future], dim=2)
    # future_feats: [B, N_agent, f, 256]
    # future_anchors: [B, N_agent, f, 11]

    # 复用 baseline motion predictor 的 decoder layers
    instance_feat = I_0["agent"]["feature"]
    anchor_embed = self.anchor_encoder(I_0["agent"]["anchor"])

    for layer in self.motion_layers:
        # temp_gnn: cross-attention to FUTURE instances (替代原来的历史 instances)
        instance_feat = layer.temp_gnn(
            query = instance_feat,
            key   = future_feats.flatten(0,1),   # [B*N_agent, f, 256]
            value = future_feats.flatten(0,1),
            query_pos = anchor_embed,
            key_pos = self.anchor_encoder(future_anchors.flatten(0,1)),
        )
        # gnn: self-attention
        instance_feat = layer.gnn(instance_feat, anchor_embed)
        # cross_gnn: cross-attention with map
        instance_feat = layer.cross_gnn(instance_feat, map_features)
        # ffn
        instance_feat = layer.ffn(instance_feat)

    # refine
    motion_query = motion_mode_query + instance_feat
    _, tau_refined = self.refine_module.motion_branch(motion_query)
    # tau_refined: [B, N_agent, K_m, T_m, 2]

    return tau_refined


def refine_planning(self, ego_feature, future_features):
    """
    Trajectory Planning Refinement
    Ego feature 与未来实例特征做 cross-attention, 生成精炼轨迹
    paper: Sec III-C "Trajectory Planning Refinement"

    Input:
      ego_feature: [B, 1, 256]
      future_features: list of [B, N_agent, 256] × f

    Output:
      tau_refined_ego: [B, K_p, T_p, 2]
    """
    # 拼接未来帧的 agent features
    f_agent_feats = torch.stack(future_features, dim=2)  # [B, N_agent, f, 256]

    # Cross-attention: ego 看未来 agents
    enhanced_ego = self.ego_future_cross_attn(
        query = ego_feature,                 # [B, 1, 256]
        key   = f_agent_feats.flatten(1,2),  # [B, N_agent*f, 256]
        value = f_agent_feats.flatten(1,2),
    )

    # 通过 planning head 生成精炼轨迹
    plan_query = self.plan_mode_query + enhanced_ego  # [B, K_p, 256]
    _, tau_refined = self.refine_module.plan_branch(plan_query)
    # tau_refined: [B, K_p, T_p, 2]
    return tau_refined
```

---

## 6. Safety-Critical Loss (SCL)

```python
# ============================================================
# 文件: models/motion/safety_loss.py
# 类: SafetyCriticalLoss
# paper Eq.(5)-(6) & Sec III-C.1
# ============================================================

class SafetyCriticalLoss(nn.Module):
    """
    SCL: 比 VAD 的碰撞约束更丰富 — 考虑未来帧 agent 位置 + agent geometry
    paper: Eq.(5) v_adj = SAV(τ, a_agent, τ_agent, θ)
    paper: Eq.(6) L_scl = mean(||v_adj_t||_2 for nonzero entries)
    """
    def __init__(self, safety_distance=2.0, ego_dims=(4.08, 1.73, 1.56)):
        """
        Args:
          safety_distance: θ (paper未明确，推测2m)
          ego_dims: (length, width, height) of ego vehicle
        """
        self.theta = safety_distance
        self.ego_l, self.ego_w, self.ego_h = ego_dims

    def forward(self, ego_trajectory, agent_anchors, agent_trajectories):
        """
        Input:
          ego_trajectory: [B, T, 2] — ego planning trajectory
          agent_anchors: [B, N, 11] — agent bounding boxes
          agent_trajectories: [B, N, T, 2] — agent predicted trajectories

        Output:
          L_scl: scalar loss
        """
        B, T, _ = ego_trajectory.shape
        N = agent_anchors.shape[1]

        # ── 每个 future timestep 计算 ego 和 agent 的 2D bounding boxes ──
        # ego box: 由 ego_trajectory_t + ego_dims 构造
        ego_boxes = self._ego_traj_to_boxes(ego_trajectory)  # [B, T, 5]

        # agent boxes: 由 agent_anchor + agent_trajectory_t 构造
        agent_boxes = self._agent_traj_to_boxes(
            agent_anchors, agent_trajectories
        )  # [B, N, T, 5]

        # ── 计算 adjustment vector v_adj (paper Eq.5) ──
        v_adj = self._compute_adjustment_vector(
            ego_boxes, agent_boxes
        )  # [B, T, 2]

        # ── 仅对非零 entries 计算 L2 norm (paper Eq.6) ──
        mask = (v_adj.norm(dim=-1) > 0).float()  # [B, T]

        # 如果所有 frame 都无碰撞，loss = 0
        if mask.sum() == 0:
            return ego_trajectory.new_zeros(1).mean()

        L_scl = (v_adj.norm(dim=-1) * mask).sum() / mask.sum()
        return L_scl

    def _compute_adjustment_vector(self, ego_boxes, agent_boxes):
        """
        计算将 ego 从碰撞状态推出的最小位移向量
        ego_boxes: [B, T, 5] (x, y, w, l, yaw)
        agent_boxes: [B, N, T, 5]
        Returns: v_adj [B, T, 2]
        """
        # 对每个 timestep, 找到与 ego 重叠的 agent boxes
        # 计算 minimal 位移使 ego 脱离 overlap (考虑安全距离 θ)
        ...
        return v_adj
```

---

## 7. Adaptive Trajectory Selection (ATS)

```python
# ============================================================
# 文件: models/motion/adaptive_trajectory_selection.py
# 类: AdaptiveTrajectorySelection
# paper: Sec III-C.2
# ============================================================

class AdaptiveTrajectorySelection(nn.Module):
    """
    ATS: 从 3 个候选轨迹中选最安全的
    候选: (1) τ_base — baseline 输出
          (2) τ_future — world model 预测
          (3) τ_refined — SCL 精炼后
    选择: SCL 最低且无碰撞的轨迹
    最终: 应用 adjustment vector
    """
    def __init__(self, safety_distance=2.0):
        self.theta = safety_distance

    def forward(self, candidates, agent_predictions, future_agent_boxes):
        """
        Input:
          candidates: List[Tuple[str, tensor]]
            [("base",    tau_base,    [B, T_p, 2]),
             ("future",  tau_future,  [B, T_p, 2]),
             ("refined", tau_refined, [B, T_p, 2])]
          agent_predictions: [B, N, T_p, 2] or [B, N, K_m, T_m, 2]
          future_agent_boxes: [B, N, 11] × f frames

        Output:
          final_trajectory: [B, T_p, 2]
          selection_info: Dict with metadata
        """
        B = candidates[0][1].shape[0]
        T_p = candidates[0][1].shape[1]

        # ── 对每条候选轨迹: 碰撞检测 + SCL 计算 ──
        scores = []
        collision_flags = []
        for name, tau in candidates:
            # 碰撞检测 (复用 detection3d/decoder.py 的 check_collision)
            has_collision = self._detect_collision(
                tau, future_agent_boxes, agent_predictions
            )  # bool

            # SCL
            scl = self._compute_scl(tau, future_agent_boxes, agent_predictions)

            scores.append(scl)
            collision_flags.append(has_collision)

        # ── 选择逻辑 ──
        # priority 1: 无碰撞
        safe_indices = [i for i, col in enumerate(collision_flags) if not col]

        if len(safe_indices) > 0:
            # 在安全候选中选 SCL 最低的
            best_idx = safe_indices[min(range(len(safe_indices)),
                                        key=lambda j: scores[safe_indices[j]])]
        else:
            # 所有候选都有碰撞 — 选 SCL 最低的 (fallback)
            best_idx = min(range(len(scores)), key=lambda i: scores[i])

        tau_selected = candidates[best_idx][1]

        # ── Apply adjustment vector ──
        # 用 selected trajectory 的 adjustment vector 修正最终轨迹
        v_adj = self._compute_adjustment_vector(
            tau_selected, future_agent_boxes, agent_predictions
        )
        tau_final = tau_selected + v_adj  # [B, T_p, 2]

        return tau_final, {
            "selected": candidates[best_idx][0],
            "collision_free": len(safe_indices) > 0,
            "scoring": dict(zip([c[0] for c in candidates], scores)),
        }
```

---

## 8. 训练管线

```python
# ============================================================
# 训练: SparseWorld 的 loss 组成
# ============================================================

def train_step_sparseworld(frame_t, frame_future_gt, is_teacher_forcing=True):
    """
    单个训练步

    假设三层训练:
      Stage 1: SparseDrive perception only (不变)
      Stage 2: SparseDrive + motion/planning (不变)
      Stage 3: Freeze perception, 训练 SparseDreamer + Refinement
               或 Joint training (paper未明确)
    """

    # ── 1. Baseline forward ──
    det_output, map_output, motion_output, planning_output = forward(frame_t)

    # ── 2. SparseDreamer 未来预测 ──
    if is_teacher_forcing:
        # Teacher forcing: 使用 GT action conditions
        C_gt = frame_future_gt["action_conditions"]
        I_future_pred = []
        I_current = I_0
        for k in range(f):
            I_next = sparse_dreamer(I_current, C_gt[k])
            I_future_pred.append(I_next)
            I_current = I_next
    else:
        # Scheduled sampling: 混合 GT 和预测
        ...

    # ── 3. Future 预测 loss ──
    L_future_det = 0
    L_future_map = 0
    for k, I_pred in enumerate(I_future_pred):
        # 对比预测的未来实例 vs GT 未来帧标注
        L_future_det += det_loss(
            I_pred["agent"]["anchor"],     # [B,900,11]
            frame_future_gt[k]["gt_bboxes_3d"],
            frame_future_gt[k]["gt_labels_3d"],
        )
        L_future_map += map_loss(
            I_pred["map"]["anchor"],        # [B,100,40]
            frame_future_gt[k]["gt_map_pts"],
        )

    # ── 4. Refinement loss ──
    tau_refined_agents = refine_motion(I_0, I_future_pred)
    tau_refined_ego = refine_planning(I_0["ego"]["feature"],
                                      [I["agent"]["feature"] for I in I_future_pred])

    L_refined_motion = motion_loss(tau_refined_agents, frame_t.gt_agent_fut_trajs)
    L_refined_plan = plan_loss(tau_refined_ego, frame_t.gt_ego_fut_trajs)

    # ── 5. Safety-Critical Loss ──
    L_scl = scl_loss(tau_refined_ego, agent_anchors, tau_refined_agents)

    # ── 6. Total loss ──
    L_total = (
        L_det + L_map + L_motion + L_plan + L_depth           # baseline losses
        + λ_future_det * L_future_det                         # new: future forecasting
        + λ_future_map * L_future_map
        + λ_refined_motion * L_refined_motion                 # new: refined motion
        + λ_refined_plan * L_refined_plan                     # new: refined planning
        + λ_scl * L_scl                                       # new: safety-critical
    )
    return L_total
```

---

## 9. 推理管线

```python
# ============================================================
# 推理: SparseWorld 全自回归 rollout + ATS
# ============================================================

@torch.no_grad()
def inference_sparseworld(frame_t):
    """
    推理单帧, 返回最终规划轨迹
    """
    # ── 1. Baseline 前向 ──
    det_output, map_output, motion_output, planning_output = forward(frame_t)

    # 初始化 IMQ 中的历史实例 (前 4 帧缓存)
    I_0 = build_current_instances(det_output, map_output, ego_feature)
    C_0 = extract_action_conditions(planning_output, ego_status)
    tau_base = planning_output["final_planning"]

    # ── 2. 自回归 rollout (f=4 steps) ──
    I_future = []
    I_current = I_0
    C_current = C_0
    for k in range(4):
        C_k = fmp(I_current)                    # FMP 推断 action conditions
        I_next = sparse_dreamer(I_current, C_k) # SparseDreamer 预测下一帧
        I_future.append(I_next)
        I_current = I_next

    # ── 3. Motion Planning Refinement ──
    tau_agents = refine_motion(I_0, I_future)
    tau_ego = refine_planning(I_0["ego"]["feature"],
                              [I["agent"]["feature"] for I in I_future])

    # Future trajectory (from FMP's last step planning output)
    tau_future = fmp.get_last_planning()

    # ── 4. Adaptive Trajectory Selection ──
    tau_final, selection_info = ats_select(
        candidates = [
            ("base",    tau_base),
            ("future",  tau_future),
            ("refined", tau_ego),
        ],
        agent_predictions = tau_agents,
        future_agent_boxes = [I["agent"]["anchor"] for I in I_future],
    )
    return tau_final
```

---

## 10. 关键维度总结

| 符号 | Shape | 含义 |
|------|-------|------|
| `feature_maps` | `[B,6,256,Hi,Wi] × 4 scales` | 多视图多尺度特征 |
| `det_output["instance_feature"]` | `[B,900,256]` | 检测实例特征 |
| `det_output["prediction"][i]` | `[B,900,11]` | 第 i 层 box delta (X,Y,Z,lnW,lnL,lnH,sin,cos,VX,VY,VZ) |
| `det_output["classification"][i]` | `[B,900,10]` | 第 i 层分类 logits |
| `det_output["instance_id"]` | `[B,900]` | 追踪 ID (-1=背景) |
| `map_output["instance_feature"]` | `[B,100,256]` | 地图实例特征 |
| `map_output["prediction"][i]` | `[B,100,40]` | 20 points × 2 coords |
| `ego_feature` (from InstanceQueue) | `[B,1,256]` | Ego vehicle 特征 (front camera pooling) |
| `ego_anchor` | `[B,1,11]` | Ego anchor (4.08×1.73×1.56, 固定) |
| `temp_instance_feature` | `[B,N,4,256]` | 历史/未来 temporal 实例特征 |
| `motion_anchor` | `[B,N,6,1,2]` | Motion intention points |
| `plan_anchor` | `[B,1,18,6,2]` | Plan intention points (3cmd × 6modes) |
| `motion_prediction` | `[B,N,6,12,2]` | 6 modes × 12 timesteps × (x,y) cumsum |
| `planning_prediction` | `[B,1,18,6,2]` | 3cmd × 6modes × 6 timesteps × (x,y) cumsum |
| `ego_status` | `[B,10]` | vx,vy,vz,ax,ay,az,ω,steer,… |
| `I_k["agent"]["anchor"]` | `[B,900,11]` | 未来帧 agent anchor (SparseDreamer output) |
| `I_k["agent"]["feature"]` | `[B,900,256]` | 未来帧 agent feature |
| `I_k["map"]["anchor"]` | `[B,100,40]` | 未来帧 map polyline |
| `C_k["velocity"]` | `[B]` | Ego velocity scalar |
| `C_k["trajectory"]` | `[B,12,2]` | Planned trajectory |
| `C_k["command"]` | `[B]` | 0=left, 1=straight, 2=right |
| `E_time` | `[m,256]` | 可学习 temporal embedding |
| `E_pos` | `[B,N,m,256]` | Relative positional encoding (Δx,Δy) |
| `action_emb` | `[B,256]` | Fourier-encoded action conditions |
| `v_adj` | `[B,T,2]` | Safety-critical adjustment vector |

---

## 11. 配置参数 (需要新增的 config 字段)

```python
# 新增 config: sparsedrive_small_stage3_sparseworld.py
# (基于 sparsedrive_small_stage2.py)

model = dict(
    # ... 原有配置 ...

    head=dict(
        # ... 原有配置 ...

        # ── 新增 SparseDreamer ──
        sparse_dreamer=dict(
            type="SparseDreamer",
            embed_dims=256,
            num_decoder=6,           # N (paper未明确, 推测=6)
            num_heads=8,
            queue_length=4,          # m (IMQ 中历史帧数)
            future_frames=4,         # f (预测未来帧数)
            decouple_attn=False,
            temp_graph_model=dict(   # temporal-cross-attn
                type="MultiheadFlashAttention",
                embed_dims=256,
                num_heads=8,
            ),
            action_graph_model=dict( # action-cross-attn (可选)
                type="MultiheadFlashAttention",
                embed_dims=256,
                num_heads=8,
            ),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=512,
                embed_dims=256,
                feedforward_channels=1024,
            ),
            agent_refine=dict(       # agent 精炼模块 (复用)
                type="SparseBox3DRefinementModule",
                embed_dims=256,
                num_cls=10,
            ),
            map_refine=dict(         # map 精炼模块 (复用)
                type="SparsePoint3DRefinementModule",
                embed_dims=256,
                num_sample=20,
                num_cls=3,
            ),
            # GIA 参数
            gia=dict(
                safety_distance=2.0, # θ for SCL
            ),
        ),

        # ── 新增 SCL ──
        safety_loss=dict(
            type="SafetyCriticalLoss",
            safety_distance=2.0,
            ego_dims=(4.08, 1.73, 1.56),
            loss_weight=1.0,         # λ_scl
        ),

        # ── 新增 ATS ──
        ats=dict(
            type="AdaptiveTrajectorySelection",
            safety_distance=2.0,
        ),

        # ── 新增 loss weights ──
        future_det_loss_weight=1.0,  # λ_future_det
        future_map_loss_weight=1.0,  # λ_future_map
        refined_motion_loss_weight=0.2,
        refined_plan_loss_weight=1.0,
    ),
)
```

---

## 12. 对 Paper 潜在问题的反思

1. **FMP 权重共享问题**: Paper 说 "identical network architecture"，但没说权重是否共享。如果是共享权重，则 FMP 无需单独训练，直接用 baseline motion planner 的权重。这在代码中体现为 `fmp = copy.deepcopy(self.motion_plan_head)` 或直接复用 `self.motion_plan_head` (去掉 InstanceQueue 依赖)。但 baseline motion planner 接受 `InstanceQueue.get()` 的输出，而 FMP 直接接受 SparseDreamer 输出 — **接口不兼容，可能需要单独实例**。

2. **训练 teacher forcing 的 GT 来源**: SparseDreamer 的 loss 需要未来帧的 GT。nuScenes 提供了 12 帧 (6s) 的 agent trajectory GT，但**没有**未来帧的 3D box GT (仅 trajectory, 不是 box)。需要从 trajectory GT 反推未来帧 boxes。Map 的未来帧 GT 更难 — map elements 是静态的，只需用未来 ego pose 变换当前 map 即可。

3. **SCL 的 adjustment vector 在推理中的应用**: Paper 说 ATS "applies the corresponding adjustment vector to the selected trajectory"。这意味着推理时需要实际修改轨迹 (加 v_adj)。这可能是**硬约束位移**，需要验证轨迹修改后的平滑性。

4. **L2 error tradeoff**: SparseWorld-S 的 L2 Avg 从 0.61 → 0.65 (+6.6%)。Coll. Rate 从 0.08% → 0.05%。这是一个安全-精度权衡，**SparseWorld 不是纯粹的"更好"，而是"更安全但略不准"**。

---

*文档生成时间: 2025-06-17*
*代码验证: SparseDrive@ec0225d, motion_planning_head.py(483行), instance_queue.py(213行)*
*Paper: SparseWorld arxiv:2605.24394 (May 2026, ZJU + Huawei)*
