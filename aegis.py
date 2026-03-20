import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GraphNorm, SAGEConv
from torch_geometric.utils import softmax, degree, dropout_edge
import math


# ==========================================
# 0. 基础组件: DropPath (随机深度)
# ==========================================
class DropPath(nn.Module):
    """Stochastic Depth: 在训练时随机丢弃残差路径，增强泛化能力"""

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
# 1. 空间组件 A: 宏观拓扑提取器 (Macro-Topology GNN)
# 抛弃边特征，纯粹利用图的连通性捕捉宏观异常拓扑 (如 DDoS, 端口扫描)
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
        out = self.dropout(out)
        return out + residual


# ==========================================
# 2. 空间组件 B: 微观交互提取器 (Micro-Interaction GNN)
# 包含边增强注意力和边更新，深挖细粒度的流量行为语义 (如 APT, SQL注入)
# ==========================================
class EdgeAugmentedAttention(MessagePassing):
    def __init__(self, in_dim, out_dim, edge_dim, heads=4, dropout=0.1, drop_path=0.1):
        super().__init__(node_dim=0, aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.heads = heads
        self.head_dim = out_dim // heads
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
        out = out.view(-1, self.out_dim)
        out = self.out_proj(out)
        out = self.norm(out + self.drop_path(residual), batch)
        return self.act(out)

    def message(self, q_i, k_j, v_j, e_emb, index):
        score = (q_i * (k_j + e_emb)).sum(dim=-1) / (self.head_dim ** 0.5)
        alpha = softmax(score, index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
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
        update = self.mlp(cat_feat)
        gate = self.hetero_gate(cat_feat)
        return self.norm(update * gate + edge_attr)


# ==========================================
# 3. 空间门控融合: 基于图结构熵的双空间解耦融合
# ==========================================
class SpatialEntropyGating(nn.Module):
    """
    ✨ 利用宏观图结构熵，动态决定信任 Macro-GNN (拓扑) 还是 Micro-GNN (特征)。
    """
    # ✅ 这里只有 hidden_dim，绝对没有 in_channels 或 out_channels
    def __init__(self, hidden_dim):
        super().__init__()
        # SiLU (Swish) 保留负梯度，平滑融合边界
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x_macro, x_micro, graph_entropy):
        """
        x_macro/x_micro: [Num_Nodes, Hidden]
        graph_entropy: [1] (当前帧的全局标量熵)
        """
        N = x_macro.size(0)
        # 加上 .view(1, 1) 防止 0-d 展开报错
        entropy_expanded = graph_entropy.view(1, 1).expand(N, 1)

        gate_input = torch.cat([x_macro, x_micro, entropy_expanded], dim=-1)
        alpha = self.gate(gate_input)  # [N, 1]

        # 融合机制：α*Macro + (1-α)*Micro
        out = alpha * x_macro + (1 - alpha) * x_micro
        return out, alpha


# ==========================================
# 4. 统一时序组件: 熵调节自适应多尺度卷积 (ER-SKNet)
# ==========================================
class AdaptiveTemporalInception(nn.Module):
    """
    ✨ 核心创新：熵调节的自适应多尺度时序卷积 (Entropy-Regulated Selective Kernel)
    模型根据输入序列特征和全图结构熵，自适应地动态融合不同大小感受野的卷积核输出。
    """
    # ✅ 这里才是接收 in_channels, out_channels 和 kernels 的地方！
    def __init__(self, in_channels, out_channels, kernels=None):
        super().__init__()
        # 接收外部传入的池子，或者默认全覆盖
        self.kernels = kernels if kernels is not None else [1, 3, 5, 7, 9, 11]

        # 多尺度卷积分支 (使用 BatchNorm1d 兼容动态序列长度)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_channels),
                nn.GELU()
            ) for k in self.kernels
        ])

        # 1D 全局平均池化，提取序列的时序特征摘要
        self.gap = nn.AdaptiveAvgPool1d(1)

        # 自适应核选择的注意力网络 (Gate)
        self.fc1 = nn.Linear(out_channels + 1, out_channels // 2)
        self.act = nn.SiLU()  # Swish 函数
        self.fc2 = nn.Linear(out_channels // 2, len(self.kernels))

        self.project = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.final_act = nn.GELU()

    def forward(self, x, graph_entropy):
        B, C, T = x.shape

        # 1. 多尺度特征提取
        U = [conv(x) for conv in self.convs]
        U_stack = torch.stack(U, dim=1)  # [B, Num_Kernels, C, T]

        # 2. 全局信息融合 (Global Information Extraction)
        U_sum = U_stack.sum(dim=1)  # [B, C, T]
        s = self.gap(U_sum).squeeze(-1)  # [B, C]

        # 3. ✨ 注入图结构熵，计算动态感受野权重
        z = torch.cat([s, graph_entropy], dim=-1)  # [B, C+1]
        attn_scores = self.fc2(self.act(self.fc1(z)))  # [B, Num_Kernels]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, Num_Kernels]

        # 4. 动态软融合 (Adaptive Soft Fusion)
        attn_weights_view = attn_weights.view(B, len(self.kernels), 1, 1)
        V = (U_stack * attn_weights_view).sum(dim=1)  # [B, C, T]

        # 残差连接
        out = V + self.project(x)
        return self.final_act(out), attn_weights


# ==========================================
# 5. 完整模型: AEGIS (Adaptive Entropy-Guided Intrusion Shield)
# ==========================================
class AEGIS(nn.Module):
    def __init__(self, node_in, edge_in, hidden, num_classes, seq_len=10, heads=8, dropout=0.3, max_cl_edges=2048,
                 drop_path=0.1, dropedge_p=0.2, kernels=None, **kwargs):
        super(AEGIS, self).__init__()
        self.hidden = hidden
        self.seq_len = seq_len
        self.max_cl_edges = max_cl_edges
        self.dropedge_p = float(dropedge_p)

        # 自适应特征衰减因子
        self.decay_factor = nn.Parameter(torch.tensor(0.8))

        # --- Encoders ---
        self.node_enc = nn.Sequential(nn.Linear(node_in, hidden), nn.LayerNorm(hidden))
        self.edge_enc = nn.Sequential(nn.Linear(edge_in, hidden), nn.LayerNorm(hidden))

        # --- Phase 1: 双粒度空间解耦层 (Dual-Granularity Spatial Evolution) ---
        self.num_layers = 2
        self.macro_spatial_layers = nn.ModuleList()
        self.micro_spatial_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.macro_spatial_layers.append(MacroTopologyGNN(hidden, dropout))
            self.micro_spatial_layers.append(nn.ModuleDict({
                'node_att': EdgeAugmentedAttention(hidden, hidden, hidden, heads, dropout, drop_path=float(drop_path)),
                'edge_upd': EdgeUpdaterModule(hidden, hidden, hidden, dropout)
            }))

        self.spatial_gating = SpatialEntropyGating(hidden)

        # --- Phase 3: 统一自适应时序层 (Adaptive Temporal Inception) ---
        self.tpe = nn.Embedding(seq_len, hidden)
        self.stream_temporal = AdaptiveTemporalInception(hidden, hidden, kernels=kernels)

        # --- Phase 4 & 5: Readout & Contrastive Head ---
        self.proj_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.reconstruct_head = nn.Sequential(nn.Linear(hidden * 3, hidden * 2), nn.ReLU(),
                                              nn.Linear(hidden * 2, hidden))
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2), nn.LayerNorm(hidden * 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, num_classes)
        )

    # 结构熵计算
    def compute_structural_entropy(self, edge_index, num_nodes):
        deg = degree(edge_index[0], num_nodes, dtype=torch.float)
        p_i = 1.0 / (deg[edge_index[0]] + 1e-6)
        entropy_edge = - p_i * torch.log(p_i + 1e-6)
        return entropy_edge

    def forward(self, graphs):
        spatial_node_feats, spatial_edge_feats = [], []
        active_edge_indices, edge_masks = [], []
        batch_global_ids, batch_graph_entropies = [], []

        # === Phase 1: Dual-Granularity Spatial Evolution ===
        def _spatial_encode_one_frame(data, dropedge_p):
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            batch = data.batch if hasattr(data, "batch") else None
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)

            frame_global_ids = data.n_id if hasattr(data, "n_id") else torch.arange(x.size(0), device=x.device)

            edge_entropy = self.compute_structural_entropy(edge_index, x.size(0))
            graph_entropy_scalar = edge_entropy.mean() if edge_index.size(1) > 0 else torch.tensor(0.0, device=x.device)

            if self.training and float(dropedge_p) > 0.0 and edge_index.size(1) > 0:
                norm_entropy = (edge_entropy - edge_entropy.min()) / (edge_entropy.max() - edge_entropy.min() + 1e-6)
                keep_prob = 1.0 - (float(dropedge_p) * norm_entropy)
                edge_mask = (keep_prob + torch.rand_like(keep_prob)).floor().bool()
                edge_index_d, edge_attr_d = edge_index[:, edge_mask], edge_attr[edge_mask]
            else:
                edge_index_d, edge_attr_d = edge_index, edge_attr
                edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)

            x_base = self.node_enc(x)
            e_micro = self.edge_enc(edge_attr_d)

            # 🌀 Stream 1: Macro-Topology (Pure Structural)
            x_macro = x_base
            for layer in self.macro_spatial_layers:
                x_macro = layer(x_macro, edge_index_d, batch)

            # 🌀 Stream 2: Micro-Interaction (Feature Intensive)
            x_micro = x_base
            for layer in self.micro_spatial_layers:
                x_micro = layer["node_att"](x_micro, edge_index_d, e_micro, batch)
                e_micro = layer["edge_upd"](x_micro, edge_index_d, e_micro)

            # 🌀 结构熵动态门控融合
            x_fused, _ = self.spatial_gating(x_macro, x_micro, graph_entropy_scalar)

            return x_fused, e_micro, edge_index_d.clone(), edge_mask, frame_global_ids, graph_entropy_scalar

        for t in range(self.seq_len):
            x, edge_feat, edge_idx_act, e_mask, frame_ids, g_entropy = _spatial_encode_one_frame(graphs[t],
                                                                                                 self.dropedge_p)
            batch_global_ids.append(frame_ids)
            edge_masks.append(e_mask)
            active_edge_indices.append(edge_idx_act)
            spatial_node_feats.append(x)
            spatial_edge_feats.append(edge_feat)
            batch_graph_entropies.append(g_entropy)

        # === Phase 2: Dynamic Alignment ===
        all_ids = torch.cat(batch_global_ids)
        unique_ids, _ = torch.sort(torch.unique(all_ids))
        num_unique = len(unique_ids)
        device = unique_ids.device

        dense_stack = torch.zeros((num_unique, self.seq_len, self.hidden), device=device)
        presence_mask = torch.zeros((num_unique, self.seq_len), device=device, dtype=torch.bool)

        for t in range(self.seq_len):
            indices = torch.searchsorted(unique_ids, batch_global_ids[t])
            dense_stack[indices, t, :] = spatial_node_feats[t]
            presence_mask[indices, t] = True

        decay_weight = torch.sigmoid(self.decay_factor)
        for t in range(1, self.seq_len):
            missing_nodes = ~presence_mask[:, t]
            dense_stack[missing_nodes, t, :] = dense_stack[missing_nodes, t - 1, :].clone() * decay_weight

        # === Phase 3: 熵调节自适应多尺度时序处理 (ER-SKNet) ===
        t_emb = self.tpe(torch.arange(self.seq_len, device=device)).unsqueeze(0)
        x_temporal_in = dense_stack + t_emb

        graph_entropies = torch.stack(batch_graph_entropies)
        # 求均值并广播至所有节点: [Num_Unique_Nodes, 1]
        mean_graph_entropy = graph_entropies.mean().unsqueeze(0).expand(num_unique, 1)

        # 转换为 Conv1d 需要的维度: [Batch, Channel, Time]
        x_temporal_in_permuted = x_temporal_in.permute(0, 2, 1)

        # ✨ 单一自适应时序流，返回特征和核权重(供可视化分析)
        x_temporal_out, kernel_weights = self.stream_temporal(x_temporal_in_permuted, mean_graph_entropy)

        # 变回 [Batch, Time, Channel]
        dense_out = x_temporal_out.permute(0, 2, 1)

        # === Phase 4 & 5: Readout & Regulated Contrastive Learning ===
        batch_preds = []
        cl_loss = torch.tensor(0.0, device=device)

        for t in range(self.seq_len):
            indices = torch.searchsorted(unique_ids, batch_global_ids[t])
            node_out_t = dense_out[indices, t, :]

            src, dst = active_edge_indices[t][0], active_edge_indices[t][1]
            edge_rep = torch.cat([spatial_edge_feats[t], node_out_t[src], node_out_t[dst]], dim=1)

            batch_preds.append(self.classifier(edge_rep))

            if self.training and t == self.seq_len // 2:
                edge_feat_anchor = spatial_edge_feats[t]
                if edge_feat_anchor is not None and edge_feat_anchor.size(0) > 0:
                    if edge_feat_anchor.size(0) > self.max_cl_edges:
                        perm = torch.randperm(edge_feat_anchor.size(0), device=device)[: self.max_cl_edges]
                        edge_feat_anchor = edge_feat_anchor[perm]
                        edge_rep_sampled = edge_rep[perm]
                    else:
                        edge_rep_sampled = edge_rep

                    z1 = F.normalize(self.reconstruct_head(edge_rep_sampled), dim=1)
                    z2 = F.normalize(self.proj_head(edge_feat_anchor), dim=1)
                    logits = torch.matmul(z1, z2.T) / 0.1
                    labels = torch.arange(z1.size(0), device=device)
                    base_cl_loss = F.cross_entropy(logits, labels)

                    # ✨ 熵正则化 CL Loss 放大：利用当前帧局部熵动态控制泛化目标
                    dynamic_cl_scale = 1.0 + torch.tanh(batch_graph_entropies[t])
                    cl_loss = base_cl_loss * dynamic_cl_scale
                else:
                    cl_loss = torch.tensor(0.0, device=device)

        self._last_edge_masks = edge_masks

        # 如果需要，可以将 kernel_weights 存为类属性供外部评价脚本读取
        self._last_kernel_weights = kernel_weights

        return batch_preds, cl_loss