import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GraphNorm
from torch_geometric.utils import softmax, degree

# 从最新的 aegis.py 导入核心组件
from aegis import (
    DropPath,
    MacroTopologyGNN,
    EdgeAugmentedAttention,
    EdgeUpdaterModule,
    SpatialEntropyGating,
    AdaptiveTemporalInception
)

# ==========================================
# 辅助组件 1: 普通注意力 (用于 w/o Edge Aug)
# ==========================================
class NormalGraphAttention(MessagePassing):
    """普通的图注意力机制（不融合 Edge Features）"""
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1, drop_path=0.1):
        super().__init__(node_dim=0, aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.head_dim = out_dim // heads
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
        out = out.view(-1, self.out_dim)
        out = self.out_proj(out)
        out = self.norm(out + self.drop_path(residual), batch)
        return self.act(out)

    def message(self, q_i, k_j, v_j, index):
        score = (q_i * k_j).sum(dim=-1) / (self.head_dim ** 0.5)
        alpha = softmax(score, index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha.unsqueeze(-1) * v_j

# ==========================================
# 辅助组件 2: 固定权重时序卷积 (用于 FixedTemporal)
# ==========================================
class FixedTemporalInception(nn.Module):
    """移除基于图结构熵的自适应门控，将不同感受野的权重固定为均值"""
    def __init__(self, in_channels, out_channels, kernels=None):
        super().__init__()
        self.kernels = kernels if kernels is not None else [1, 3, 5, 7, 9, 11]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(out_channels), 
                nn.GELU()
            ) for k in self.kernels
        ])
        self.project = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.final_act = nn.GELU()

    def forward(self, x, graph_entropy):
        B, C, T = x.shape
        U = [conv(x) for conv in self.convs] 
        U_stack = torch.stack(U, dim=1) # [B, Num_Kernels, C, T]
        
        # 固定赋予相等的权重 (摒弃熵自适应)
        attn_weights = torch.ones(B, len(self.kernels), device=x.device) / len(self.kernels)
        attn_weights_view = attn_weights.view(B, len(self.kernels), 1, 1)
        
        V = (U_stack * attn_weights_view).sum(dim=1) 
        out = V + self.project(x)
        return self.final_act(out), attn_weights

# ==========================================
# 基础消融模板类 (提取 AEGIS 公共逻辑)
# ==========================================
class BaseAblationAEGIS(nn.Module):
    def __init__(
        self, node_in, edge_in, hidden, num_classes,
        seq_len=10, heads=8, dropout=0.3, max_cl_edges=2048,
        kernels=None, drop_path=0.1, dropedge_p=0.2
    ):
        super().__init__()
        self.hidden = hidden
        self.seq_len = seq_len
        self.max_cl_edges = max_cl_edges
        self.dropedge_p = float(dropedge_p)
        self.decay_factor = nn.Parameter(torch.tensor(0.8))
        
        self.node_enc = nn.Sequential(nn.Linear(node_in, hidden), nn.LayerNorm(hidden))
        self.edge_enc = nn.Sequential(nn.Linear(edge_in, hidden), nn.LayerNorm(hidden))
        
        self.tpe = nn.Embedding(seq_len, hidden)
        
        self.proj_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.reconstruct_head = nn.Sequential(nn.Linear(hidden * 3, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden))
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2), nn.LayerNorm(hidden * 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, num_classes)
        )

    def compute_structural_entropy(self, edge_index, num_nodes):
        deg = degree(edge_index[0], num_nodes, dtype=torch.float)
        p_i = 1.0 / (deg[edge_index[0]] + 1e-6)
        return - p_i * torch.log(p_i + 1e-6)

    def _prepare_spatial_data(self, data, dropedge_p):
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
        e_base = self.edge_enc(edge_attr_d)
        return x_base, e_base, edge_index_d, edge_mask, batch, frame_global_ids, graph_entropy_scalar

    def _spatial_encode_one_frame(self, data, dropedge_p):
        raise NotImplementedError("Subclasses must implement spatial encoding.")

    def _align_temporal_features(self, batch_global_ids, spatial_node_feats):
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
            dense_stack[missing_nodes, t, :] = dense_stack[missing_nodes, t-1, :].clone() * decay_weight
            
        return dense_stack, unique_ids, num_unique

    def _readout_and_classify(self, dense_out, batch_global_ids, unique_ids, active_edge_indices, spatial_edge_feats, batch_graph_entropies):
        batch_preds = []
        cl_loss = torch.tensor(0.0, device=dense_out.device)
        device = dense_out.device
        
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
                    
                    dynamic_cl_scale = 1.0 + torch.tanh(batch_graph_entropies[t])
                    cl_loss = base_cl_loss * dynamic_cl_scale
                else:
                    cl_loss = torch.tensor(0.0, device=device)

        return batch_preds, cl_loss

    def forward(self, graphs):
        spatial_node_feats, spatial_edge_feats = [], []
        active_edge_indices, edge_masks = [], []
        batch_global_ids, batch_graph_entropies = [], []

        for t in range(self.seq_len):
            x, edge_feat, edge_idx_act, e_mask, frame_ids, g_entropy = self._spatial_encode_one_frame(graphs[t], self.dropedge_p)
            batch_global_ids.append(frame_ids)
            edge_masks.append(e_mask)
            active_edge_indices.append(edge_idx_act)
            spatial_node_feats.append(x)
            spatial_edge_feats.append(edge_feat)
            batch_graph_entropies.append(g_entropy)

        dense_stack, unique_ids, num_unique = self._align_temporal_features(batch_global_ids, spatial_node_feats)
        
        t_emb = self.tpe(torch.arange(self.seq_len, device=dense_stack.device)).unsqueeze(0)
        x_temporal_in = dense_stack + t_emb
        
        mean_graph_entropy = torch.stack(batch_graph_entropies).mean().unsqueeze(0).expand(num_unique, 1)
        x_temporal_out, kernel_weights = self.stream_temporal(x_temporal_in.permute(0, 2, 1), mean_graph_entropy)
        dense_out = x_temporal_out.permute(0, 2, 1)

        batch_preds, cl_loss = self._readout_and_classify(dense_out, batch_global_ids, unique_ids, active_edge_indices, spatial_edge_feats, batch_graph_entropies)
        
        self._last_edge_masks = edge_masks
        self._last_kernel_weights = kernel_weights
        return batch_preds, cl_loss

# ==========================================
# 变体 1: w/o Macro (仅保留 Micro 边特征流)
# ==========================================
class AEGIS_WoMacro(BaseAblationAEGIS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = 2
        self.micro_spatial_layers = nn.ModuleList([
            nn.ModuleDict({
                'node_att': EdgeAugmentedAttention(self.hidden, self.hidden, self.hidden, kwargs.get('heads', 8), kwargs.get('dropout', 0.3), drop_path=kwargs.get('drop_path', 0.1)),
                'edge_upd': EdgeUpdaterModule(self.hidden, self.hidden, self.hidden, kwargs.get('dropout', 0.3))
            }) for _ in range(self.num_layers)
        ])
        self.stream_temporal = AdaptiveTemporalInception(self.hidden, self.hidden, kernels=kwargs.get('kernels'))

    def _spatial_encode_one_frame(self, data, dropedge_p):
        x_base, e_micro, edge_index_d, edge_mask, batch, frame_global_ids, graph_entropy_scalar = self._prepare_spatial_data(data, dropedge_p)
        x_micro = x_base
        for layer in self.micro_spatial_layers:
            x_micro = layer["node_att"](x_micro, edge_index_d, e_micro, batch)
            e_micro = layer["edge_upd"](x_micro, edge_index_d, e_micro)
        return x_micro, e_micro, edge_index_d.clone(), edge_mask, frame_global_ids, graph_entropy_scalar

# ==========================================
# 变体 2: w/o Micro (仅保留 Macro 拓扑连通性流)
# ==========================================
class AEGIS_WoMicro(BaseAblationAEGIS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = 2
        self.macro_spatial_layers = nn.ModuleList([MacroTopologyGNN(self.hidden, kwargs.get('dropout', 0.3)) for _ in range(self.num_layers)])
        self.stream_temporal = AdaptiveTemporalInception(self.hidden, self.hidden, kernels=kwargs.get('kernels'))

    def _spatial_encode_one_frame(self, data, dropedge_p):
        x_base, e_micro, edge_index_d, edge_mask, batch, frame_global_ids, graph_entropy_scalar = self._prepare_spatial_data(data, dropedge_p)
        x_macro = x_base
        for layer in self.macro_spatial_layers:
            x_macro = layer(x_macro, edge_index_d, batch)
        return x_macro, e_micro, edge_index_d.clone(), edge_mask, frame_global_ids, graph_entropy_scalar

# ==========================================
# 变体 3: w/o Spatial Gating (直接相加融合空间双流)
# ==========================================
class AEGIS_WoSpatialGating(BaseAblationAEGIS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = 2
        self.macro_spatial_layers = nn.ModuleList([MacroTopologyGNN(self.hidden, kwargs.get('dropout', 0.3)) for _ in range(self.num_layers)])
        self.micro_spatial_layers = nn.ModuleList([
            nn.ModuleDict({
                'node_att': EdgeAugmentedAttention(self.hidden, self.hidden, self.hidden, kwargs.get('heads', 8), kwargs.get('dropout', 0.3), drop_path=kwargs.get('drop_path', 0.1)),
                'edge_upd': EdgeUpdaterModule(self.hidden, self.hidden, self.hidden, kwargs.get('dropout', 0.3))
            }) for _ in range(self.num_layers)
        ])
        self.stream_temporal = AdaptiveTemporalInception(self.hidden, self.hidden, kernels=kwargs.get('kernels'))

    def _spatial_encode_one_frame(self, data, dropedge_p):
        x_base, e_micro, edge_index_d, edge_mask, batch, frame_global_ids, graph_entropy_scalar = self._prepare_spatial_data(data, dropedge_p)
        x_macro = x_base
        for layer in self.macro_spatial_layers:
            x_macro = layer(x_macro, edge_index_d, batch)
            
        x_micro = x_base
        for layer in self.micro_spatial_layers:
            x_micro = layer["node_att"](x_micro, edge_index_d, e_micro, batch)
            e_micro = layer["edge_upd"](x_micro, edge_index_d, e_micro)
            
        x_fused = x_macro + x_micro # 直接物理相加，无熵门控
        return x_fused, e_micro, edge_index_d.clone(), edge_mask, frame_global_ids, graph_entropy_scalar

# ==========================================
# 变体 4: w/o Edge Aug (微观流使用普通注意力机制)
# ==========================================
class AEGIS_WoEdgeAug(BaseAblationAEGIS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = 2
        self.macro_spatial_layers = nn.ModuleList([MacroTopologyGNN(self.hidden, kwargs.get('dropout', 0.3)) for _ in range(self.num_layers)])
        self.micro_spatial_layers = nn.ModuleList([
            nn.ModuleDict({
                'node_att': NormalGraphAttention(self.hidden, self.hidden, kwargs.get('heads', 8), kwargs.get('dropout', 0.3), drop_path=kwargs.get('drop_path', 0.1)),
                'edge_upd': EdgeUpdaterModule(self.hidden, self.hidden, self.hidden, kwargs.get('dropout', 0.3))
            }) for _ in range(self.num_layers)
        ])
        self.spatial_gating = SpatialEntropyGating(self.hidden)
        self.stream_temporal = AdaptiveTemporalInception(self.hidden, self.hidden, kernels=kwargs.get('kernels'))

    def _spatial_encode_one_frame(self, data, dropedge_p):
        x_base, e_micro, edge_index_d, edge_mask, batch, frame_global_ids, graph_entropy_scalar = self._prepare_spatial_data(data, dropedge_p)
        
        x_macro = x_base
        for layer in self.macro_spatial_layers:
            x_macro = layer(x_macro, edge_index_d, batch)
            
        x_micro = x_base
        for layer in self.micro_spatial_layers:
            x_micro = layer["node_att"](x_micro, edge_index_d, batch) # 未增强边缘
            e_micro = layer["edge_upd"](x_micro, edge_index_d, e_micro)
            
        x_fused, _ = self.spatial_gating(x_macro, x_micro, graph_entropy_scalar)
        return x_fused, e_micro, edge_index_d.clone(), edge_mask, frame_global_ids, graph_entropy_scalar

# ==========================================
# 变体 5: Fixed Temporal (感受野固定不变，无自适应熵)
# ==========================================
class AEGIS_FixedTemporal(BaseAblationAEGIS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = 2
        self.macro_spatial_layers = nn.ModuleList([MacroTopologyGNN(self.hidden, kwargs.get('dropout', 0.3)) for _ in range(self.num_layers)])
        self.micro_spatial_layers = nn.ModuleList([
            nn.ModuleDict({
                'node_att': EdgeAugmentedAttention(self.hidden, self.hidden, self.hidden, kwargs.get('heads', 8), kwargs.get('dropout', 0.3), drop_path=kwargs.get('drop_path', 0.1)),
                'edge_upd': EdgeUpdaterModule(self.hidden, self.hidden, self.hidden, kwargs.get('dropout', 0.3))
            }) for _ in range(self.num_layers)
        ])
        self.spatial_gating = SpatialEntropyGating(self.hidden)
        
        # 替换为固定感受野时序模块
        self.stream_temporal = FixedTemporalInception(self.hidden, self.hidden, kernels=kwargs.get('kernels'))

    def _spatial_encode_one_frame(self, data, dropedge_p):
        x_base, e_micro, edge_index_d, edge_mask, batch, frame_global_ids, graph_entropy_scalar = self._prepare_spatial_data(data, dropedge_p)
        
        x_macro = x_base
        for layer in self.macro_spatial_layers:
            x_macro = layer(x_macro, edge_index_d, batch)
            
        x_micro = x_base
        for layer in self.micro_spatial_layers:
            x_micro = layer["node_att"](x_micro, edge_index_d, e_micro, batch)
            e_micro = layer["edge_upd"](x_micro, edge_index_d, e_micro)
            
        x_fused, _ = self.spatial_gating(x_macro, x_micro, graph_entropy_scalar)
        return x_fused, e_micro, edge_index_d.clone(), edge_mask, frame_global_ids, graph_entropy_scalar