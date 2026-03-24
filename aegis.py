import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GraphNorm, SAGEConv
from torch_geometric.utils import softmax, degree

# ==========================================
# 0. 基础组件
# ==========================================
class DropPath(nn.Module):
    """注: 这里的实现类似于 Node-wise residual dropout, 用于增强节点级表征的泛化"""
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# ==========================================
# 1. 空间组件
# ==========================================
class MacroTopologyGNN(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.conv = SAGEConv(hidden_dim, hidden_dim)
        self.norm = GraphNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        residual = x
        out = self.conv(x, edge_index)
        out = self.norm(out, batch)
        out = self.act(out)
        return self.dropout(out) + residual

class NormalGraphAttention(MessagePassing):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1, drop_path=0.1):
        super().__init__(node_dim=0, aggr='add')
        if out_dim % heads != 0:
            raise ValueError(f"out_dim={out_dim} must be divisible by heads={heads}")
        self.out_dim, self.heads, self.head_dim = out_dim, heads, out_dim // heads
        self.dropout = dropout
        self.WQ = nn.Linear(in_dim, out_dim, bias=False)
        self.WK = nn.Linear(in_dim, out_dim, bias=False)
        self.WV = nn.Linear(in_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = GraphNorm(out_dim)
        self.drop_path = DropPath(drop_path)
        self.act = nn.GELU()

    def forward(self, x, edge_index, batch=None):
        residual = x
        q = self.WQ(x).view(-1, self.heads, self.head_dim)
        k = self.WK(x).view(-1, self.heads, self.head_dim)
        v = self.WV(x).view(-1, self.heads, self.head_dim)
        out = self.propagate(edge_index, q=q, k=k, v=v, size=None)
        out = self.norm(self.out_proj(out.view(-1, self.out_dim)) + self.drop_path(residual), batch)
        return self.act(out)

    def message(self, q_i, k_j, v_j, index):
        score = (q_i * k_j).sum(dim=-1) / (self.head_dim ** 0.5)
        alpha = F.dropout(softmax(score, index), p=self.dropout, training=self.training)
        return alpha.unsqueeze(-1) * v_j

class EdgeAugmentedAttention(MessagePassing):
    def __init__(self, in_dim, out_dim, edge_dim, heads=4, dropout=0.1, drop_path=0.1):
        super().__init__(node_dim=0, aggr='add')
        if out_dim % heads != 0:
            raise ValueError(f"out_dim={out_dim} must be divisible by heads={heads}")
        self.out_dim, self.heads, self.head_dim = out_dim, heads, out_dim // heads
        self.dropout = dropout
        self.WQ = nn.Linear(in_dim, out_dim, bias=False)
        self.WK = nn.Linear(in_dim, out_dim, bias=False)
        self.WV = nn.Linear(in_dim, out_dim, bias=False)
        self.WE = nn.Linear(edge_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = GraphNorm(out_dim)
        self.drop_path = DropPath(drop_path)
        self.act = nn.GELU()

    def forward(self, x, edge_index, edge_attr, batch=None):
        residual = x
        q = self.WQ(x).view(-1, self.heads, self.head_dim)
        k = self.WK(x).view(-1, self.heads, self.head_dim)
        v = self.WV(x).view(-1, self.heads, self.head_dim)
        e_emb = self.WE(edge_attr).view(-1, self.heads, self.head_dim)
        out = self.propagate(edge_index, q=q, k=k, v=v, e_emb=e_emb, size=None)
        out = self.norm(self.out_proj(out.view(-1, self.out_dim)) + self.drop_path(residual), batch)
        return self.act(out)

    def message(self, q_i, k_j, v_j, e_emb, index):
        score = (q_i * (k_j + e_emb)).sum(dim=-1) / (self.head_dim ** 0.5)
        alpha = F.dropout(softmax(score, index), p=self.dropout, training=self.training)
        return alpha.unsqueeze(-1) * (v_j + e_emb)

class EdgeUpdaterModule(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, dropout=0.1):
        super().__init__()
        input_dim = node_dim * 2 + edge_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)
        )
        self.hetero_gate = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        cat_feat = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        return self.norm(self.mlp(cat_feat) * self.hetero_gate(cat_feat) + edge_attr)

class SpatialEntropyGating(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid()
        )

    def forward(self, x_macro, x_micro, node_irregularity):
        # 使用真实的局部节点不规则度作为门控提示
        gate_input = torch.cat([x_macro, x_micro, node_irregularity.view(-1, 1)], dim=-1)
        alpha = self.gate(gate_input)
        return alpha * x_macro + (1 - alpha) * x_micro, alpha

# ==========================================
# 2. 时序组件 (含 Fixed 变体)
# ==========================================
class AdaptiveTemporalInception(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=None):
        super().__init__()
        self.kernels = kernels if kernels is not None else [1, 3, 5, 7, 9, 11]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_channels), nn.GELU()
            ) for k in self.kernels
        ])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(out_channels + 1, out_channels // 2)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(out_channels // 2, len(self.kernels))
        self.project = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.final_act = nn.GELU()

    def forward(self, x, global_irregularity):
        B, C, T = x.shape
        U = [conv(x) for conv in self.convs]
        U_stack = torch.stack(U, dim=1)
        
        s = self.gap(U_stack.sum(dim=1)).squeeze(-1)
        z = torch.cat([s, global_irregularity], dim=-1)
        attn_weights = F.softmax(self.fc2(self.act(self.fc1(z))), dim=-1)
        
        V = (U_stack * attn_weights.view(B, len(self.kernels), 1, 1)).sum(dim=1)
        return self.final_act(V + self.project(x)), attn_weights

class FixedTemporalInception(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=None):
        super().__init__()
        self.kernels = kernels if kernels is not None else [1, 3, 5, 7, 9, 11]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_channels), nn.GELU()
            ) for k in self.kernels
        ])
        self.project = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.final_act = nn.GELU()

    def forward(self, x, global_irregularity):
        B, C, T = x.shape
        U = [conv(x) for conv in self.convs]
        U_stack = torch.stack(U, dim=1)
        attn_weights = torch.ones(B, len(self.kernels), device=x.device) / len(self.kernels)
        V = (U_stack * attn_weights.view(B, len(self.kernels), 1, 1)).sum(dim=1)
        return self.final_act(V + self.project(x)), attn_weights

# ==========================================
# 3. 终极主模型: AEGIS (配置化基类)
# ==========================================
class AEGIS(nn.Module):
    def __init__(self, node_in, edge_in, hidden, num_classes, seq_len=10, heads=8, dropout=0.3, 
                 drop_path=0.1, dropedge_p=0.2, kernels=None,
                 use_macro=True, use_micro=True, use_spatial_gating=True, use_edge_aug=True, temporal_mode="adaptive", 
                 **kwargs):
        super(AEGIS, self).__init__()
        self.hidden, self.seq_len = hidden, seq_len
        self.dropedge_p = float(dropedge_p)
        
        # --- 架构控制开关 ---
        self.use_macro = use_macro
        self.use_micro = use_micro
        self.use_spatial_gating = use_spatial_gating
        self.use_edge_aug = use_edge_aug
        self.temporal_mode = temporal_mode

        self.node_enc = nn.Sequential(nn.Linear(node_in, hidden), nn.LayerNorm(hidden))
        self.edge_enc = nn.Sequential(nn.Linear(edge_in, hidden), nn.LayerNorm(hidden))

        # --- Phase 1: Spatial ---
        self.num_layers = 2
        self.macro_spatial_layers = nn.ModuleList([MacroTopologyGNN(hidden, dropout) for _ in range(self.num_layers)])
        self.micro_spatial_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            if self.use_edge_aug:
                self.micro_spatial_layers.append(nn.ModuleDict({
                    'node_att': EdgeAugmentedAttention(hidden, hidden, hidden, heads, dropout, drop_path),
                    'edge_upd': EdgeUpdaterModule(hidden, hidden, hidden, dropout)
                }))
            else:
                self.micro_spatial_layers.append(nn.ModuleDict({
                    'node_att': NormalGraphAttention(hidden, hidden, heads, dropout, drop_path),
                    'edge_upd': EdgeUpdaterModule(hidden, hidden, hidden, dropout)
                }))

        self.spatial_gating = SpatialEntropyGating(hidden) if self.use_spatial_gating else None

        # --- Phase 3: Temporal ---
        self.tpe = nn.Embedding(seq_len, hidden)
        # in_channels 为 hidden + 1，显式拼接 presence_mask
        if self.temporal_mode == "adaptive":
            self.stream_temporal = AdaptiveTemporalInception(hidden + 1, hidden, kernels=kernels)
        else:
            self.stream_temporal = FixedTemporalInception(hidden + 1, hidden, kernels=kernels)

        # --- Phase 4 & 5: Readout ---
        self.aux_proj = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 3),
            nn.LayerNorm(hidden * 3),
            nn.GELU()
        )
        self.reconstruct_head = nn.Sequential(nn.Linear(hidden * 3, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden))
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2), nn.LayerNorm(hidden * 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, num_classes)
        )

    def compute_degree_irregularity(self, edge_index, num_nodes):
        """严格对称的节点与边不规则度 (结构熵代理)"""
        src, dst = edge_index
        # 将图视作无向或严格计算总度数保证对称性
        deg_all = degree(src, num_nodes, dtype=torch.float) + degree(dst, num_nodes, dtype=torch.float)
        
        # 节点级局部不规则度
        p_node = 1.0 / (deg_all + 1e-6)
        node_irregularity = -p_node * torch.log(p_node + 1e-6)
        
        # 边级双端不规则度
        p_src, p_dst = 1.0 / (deg_all[src] + 1e-6), 1.0 / (deg_all[dst] + 1e-6)
        edge_irregularity = -0.5 * (p_src * torch.log(p_src + 1e-6) + p_dst * torch.log(p_dst + 1e-6))
        
        return node_irregularity, edge_irregularity

    def irregularity_guided_dropedge(self, edge_index, edge_attr, edge_irregularity, dropedge_p):
        """使用相对均值缩放，保证全局丢边率期望贴近 dropedge_p"""
        if not self.training or float(dropedge_p) <= 0.0 or edge_index.size(1) == 0:
            return edge_index, edge_attr, torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
            
        relative_irregularity = edge_irregularity / (edge_irregularity.mean() + 1e-6)
        drop_prob = torch.clamp(float(dropedge_p) * relative_irregularity, 0.0, 1.0)
        keep_prob = 1.0 - drop_prob
        edge_mask = torch.rand_like(keep_prob) < keep_prob 
        return edge_index[:, edge_mask], edge_attr[edge_mask], edge_mask

    def compute_aux_loss(self, edge_rep, target_feat, global_irreg_scalar, src_global_ids, dst_global_ids, batch_ids, local_mask_history):
        """
        History-aware edge masking (inspired by DVGMAE) & Latent re-masking (inspired by MGAE)
        """
        device = edge_rep.device
        feat_dim = target_feat.size(1) 
        
        # --- 1. DVGMAE-lite: 带有 Batch 隔离的跨时间窗掩码记忆 ---
        base_mask_ratio = torch.clamp(0.15 + 0.45 * torch.tanh(global_irreg_scalar), min=0.15, max=0.60)
        
        # 加入 batch_ids 组成三元组 Key，彻底隔离同一个 mini-batch 中不同子图的同名 IP
        edge_keys = list(zip(batch_ids.tolist(), src_global_ids.tolist(), dst_global_ids.tolist()))
        hist_values = torch.tensor([local_mask_history.get(k, 0.0) for k in edge_keys], device=device).unsqueeze(-1)
        
        # 软衰减：0.5 + 0.5 * (1 - hist)
        history_factor = 0.5 + 0.5 * (1.0 - hist_values)
        mask_prob_tensor = torch.clamp(base_mask_ratio * history_factor, 0.05, 0.85)

        # --- 2. 第一重 Mask: 仅遮挡 edge_feat_block ---
        mask1 = (torch.rand(edge_rep.size(0), 1, device=device) > mask_prob_tensor).float()
        edge_feat_masked = edge_rep[:, :feat_dim] * mask1
        
        edge_rep_corrupted = torch.cat([edge_feat_masked, edge_rep[:, feat_dim:]], dim=1)
        
        if self.training:
            for idx, key in enumerate(edge_keys):
                is_masked = (mask1[idx, 0].item() == 0.0) 
                old_hist = local_mask_history.get(key, 0.0)
                local_mask_history[key] = 0.8 * old_hist + 0.2 * (1.0 if is_masked else 0.0)

        # --- 3. MGAE-lite: 隐空间投影与二次掩码 ---
        # 先投影到 Latent Space，打散原始特征分布
        latent = self.aux_proj(edge_rep_corrupted)
        
        feat_part = latent[:, :feat_dim]
        ctx_part  = latent[:, feat_dim:]
        
        # 在隐空间中对边特征块做二次破坏
        re_mask = (torch.rand_like(feat_part) > 0.20).float()
        feat_part_remasked = feat_part * re_mask
        
        latent_remasked = torch.cat([feat_part_remasked, ctx_part], dim=1)

        # --- 4. 解码重构 ---
        reconstructed = self.reconstruct_head(latent_remasked)
        base_recon_loss = F.mse_loss(reconstructed, target_feat.detach())
        
        dynamic_scale = 1.0 + torch.tanh(global_irreg_scalar)
        return base_recon_loss * dynamic_scale

    def _spatial_encode_one_frame(self, data, dropedge_p):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, "batch") else None
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)

        # Stable cross-frame node identity.
        # In current dataset builders, `n_id` already stores the hashed global IP identity.
        frame_global_ids = getattr(data, "global_node_id", getattr(data, "n_id", None))
        if frame_global_ids is None:
            raise ValueError(
                "Missing stable node identity field. Expected 'global_node_id' or dataset-provided 'n_id'."
            )
        
        node_irreg, edge_irreg = self.compute_degree_irregularity(edge_index, x.size(0))
        graph_irreg_scalar = edge_irreg.mean().view(1) if edge_index.size(1) > 0 else x.new_zeros(1)

        edge_index_d, edge_attr_d, edge_mask = self.irregularity_guided_dropedge(edge_index, edge_attr, edge_irreg, dropedge_p)

        x_base = self.node_enc(x)
        e_base = self.edge_enc(edge_attr_d)

        # 🌀 Stream 1: Macro
        x_macro = x_base
        if self.use_macro:
            for layer in self.macro_spatial_layers:
                x_macro = layer(x_macro, edge_index_d, batch)
        else:
            x_macro = torch.zeros_like(x_base)

        # 🌀 Stream 2: Micro
        x_micro, e_micro = x_base, e_base
        if self.use_micro:
            for layer in self.micro_spatial_layers:
                if self.use_edge_aug:
                    x_micro = layer["node_att"](x_micro, edge_index_d, e_micro, batch)
                else:
                    x_micro = layer["node_att"](x_micro, edge_index_d, batch)
                e_micro = layer["edge_upd"](x_micro, edge_index_d, e_micro)
        else:
            # 严格消融微观消息传递，但 e_micro 保持 e_base 原值以供分类器做基础判断
            x_micro = torch.zeros_like(x_base)

        # 🌀 Fusion
        if self.use_macro and self.use_micro:
            if self.use_spatial_gating:
                x_fused, _ = self.spatial_gating(x_macro, x_micro, node_irreg)
            else:
                x_fused = 0.5 * (x_macro + x_micro) # 公平量级裸加和
        else:
            x_fused = x_macro if self.use_macro else x_micro

        return x_fused, e_micro, edge_index_d.clone(), edge_mask, frame_global_ids, graph_irreg_scalar

    def forward(self, graphs):
        spatial_node_feats, spatial_edge_feats = [], []
        active_edge_indices, edge_masks = [], []
        batch_global_ids, batch_graph_irregs = [], []

        for t in range(self.seq_len):
            x, edge_feat, edge_idx_act, e_mask, frame_ids, g_irreg = self._spatial_encode_one_frame(graphs[t], self.dropedge_p)
            batch_global_ids.append(frame_ids)
            edge_masks.append(e_mask)
            active_edge_indices.append(edge_idx_act)
            spatial_node_feats.append(x)
            spatial_edge_feats.append(edge_feat)
            batch_graph_irregs.append(g_irreg)

        # === Phase 2: Dynamic Alignment with Clean Zero-Padding ===
        all_ids = torch.cat(batch_global_ids)
        unique_ids, _ = torch.sort(torch.unique(all_ids))
        num_unique = len(unique_ids)
        device = unique_ids.device

        # 初始化为全 0，缺失节点特征自动为 0，不再做 clone() 污染
        dense_stack = torch.zeros((num_unique, self.seq_len, self.hidden), device=device)
        presence_mask = torch.zeros((num_unique, self.seq_len), device=device, dtype=torch.bool)

        for t in range(self.seq_len):
            indices = torch.searchsorted(unique_ids, batch_global_ids[t])
            dense_stack[indices, t, :] = spatial_node_feats[t]
            presence_mask[indices, t] = True

        # === Phase 3: ER-SKNet (Concatenating Presence Mask) ===
        t_emb = self.tpe(torch.arange(self.seq_len, device=device)).unsqueeze(0)
        presence_feat = presence_mask.float().unsqueeze(-1)
        # 时序 CNN 将通过 (特征=0, mask=0) 自主学习出节点缺失的语义规律
        x_temporal_in = torch.cat([dense_stack + t_emb, presence_feat], dim=-1)

        mean_graph_irreg = torch.stack(batch_graph_irregs).mean().unsqueeze(0).expand(num_unique, 1)
        x_temporal_out, kernel_weights = self.stream_temporal(x_temporal_in.permute(0, 2, 1), mean_graph_irreg)
        dense_out = x_temporal_out.permute(0, 2, 1)

        # === Phase 4 & 5: Single Frame Readout & Sequence Aux Loss ===
        aux_loss = torch.tensor(0.0, device=device)
        
        # [主任务] 仅对最后一帧分类
        t_last = self.seq_len - 1
        indices_last = torch.searchsorted(unique_ids, batch_global_ids[t_last])
        node_out_last = dense_out[indices_last, t_last, :]
        src, dst = active_edge_indices[t_last][0], active_edge_indices[t_last][1]
        
        edge_rep = torch.cat([spatial_edge_feats[t_last], node_out_last[src], node_out_last[dst]], dim=1)
        logits = self.classifier(edge_rep)

        if self.training:
            local_mask_history = {}
            aux_loss_sum = torch.tensor(0.0, device=device)
            valid_aux_steps = 0
            
            for t_aux in range(self.seq_len - 1):
                if spatial_edge_feats[t_aux].size(0) > 0:
                    indices_aux = torch.searchsorted(unique_ids, batch_global_ids[t_aux])
                    node_out_aux = dense_out[indices_aux, t_aux, :]
                    src_aux, dst_aux = active_edge_indices[t_aux][0], active_edge_indices[t_aux][1]
                    edge_rep_aux = torch.cat([spatial_edge_feats[t_aux], node_out_aux[src_aux], node_out_aux[dst_aux]], dim=1)
                    
                    src_global_ids = batch_global_ids[t_aux][src_aux]
                    dst_global_ids = batch_global_ids[t_aux][dst_aux]
                    
                    # 严密获取 Batch ID 与跨图边界断言
                    if hasattr(graphs[t_aux], "batch") and graphs[t_aux].batch is not None:
                        node_batch = graphs[t_aux].batch
                        batch_ids = node_batch[src_aux]
                        # 核心防御：一条边的源和目的节点必须属于同一个子图
                        assert torch.equal(node_batch[src_aux], node_batch[dst_aux]), "Edge crosses graph boundaries unexpectedly."
                    else:
                        batch_ids = torch.zeros(src_aux.size(0), dtype=torch.long, device=device)
                    
                    step_loss = self.compute_aux_loss(
                        edge_rep_aux, 
                        spatial_edge_feats[t_aux], 
                        batch_graph_irregs[t_aux],
                        src_global_ids,
                        dst_global_ids,
                        batch_ids,
                        local_mask_history
                    )

                    aux_loss_sum = aux_loss_sum + step_loss
                    valid_aux_steps += 1
            
            if valid_aux_steps > 0:
                aux_loss = aux_loss_sum / valid_aux_steps

        self._last_edge_masks = edge_masks
        self._last_kernel_weights = kernel_weights

        return logits, aux_loss