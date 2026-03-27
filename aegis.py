import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GraphNorm, GATv2Conv
from torch_geometric.utils import softmax, degree
from torch_scatter import scatter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ==========================================
# 0. 基础组件
# ==========================================
class DropPath(nn.Module):
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
# 1. 空间组件 (统一 Post-Norm 残差风格)
# ==========================================
class MacroTopologyGNN(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.conv = GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, concat=True)
        self.norm = GraphNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(0.1)

    def forward(self, x, edge_index, batch=None):
        residual = x
        out = self.conv(x, edge_index)
        out = self.norm(out + self.drop_path(residual), batch)
        out = self.act(out)
        return self.dropout(out)

class NormalGraphAttention(MessagePassing):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1, drop_path=0.1):
        super().__init__(node_dim=0, aggr='add')
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
        
        out = self.out_proj(out.view(-1, self.out_dim))
        out = self.norm(out + self.drop_path(residual), batch)
        return self.act(out)

    def message(self, q_i, k_j, v_j, index):
        score = (q_i * k_j).sum(dim=-1) / (self.head_dim ** 0.5)
        alpha = F.dropout(softmax(score, index), p=self.dropout, training=self.training)
        return alpha.unsqueeze(-1) * v_j

class EdgeAugmentedAttention(MessagePassing):
    def __init__(self, in_dim, out_dim, edge_dim, heads=4, dropout=0.1, drop_path=0.1):
        super().__init__(node_dim=0, aggr='add')
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
        
        out = self.out_proj(out.view(-1, self.out_dim))
        out = self.norm(out + self.drop_path(residual), batch)
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
        gate_input = torch.cat([x_macro, x_micro, node_irregularity.view(-1, 1)], dim=-1)
        alpha = self.gate(gate_input)
        return alpha * x_macro + (1 - alpha) * x_micro, alpha

# ==========================================
# 2. 时序组件
# ==========================================
class AdaptiveTemporalInception(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=None):
        super().__init__()
        self.kernels = kernels if kernels is not None else [1, 3, 5, 7, 9, 11]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2, bias=True),
                nn.GroupNorm(1, out_channels),
                nn.GELU()
            ) for k in self.kernels
        ])
        self.fc1 = nn.Linear(out_channels + 1, out_channels // 2)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(out_channels // 2, len(self.kernels))
        self.project = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.final_act = nn.GELU()

    def forward(self, x, presence_mask, node_irregularity):
        B, C, T = x.shape
        branch_outs = []
        for conv in self.convs:
            out = conv(x)
            out = out * presence_mask
            branch_outs.append(out)

        U_stack = torch.stack(branch_outs, dim=1) 
        valid_lens = presence_mask.sum(dim=-1).clamp_min(1.0)          
        s = U_stack.sum(dim=-1) / valid_lens.unsqueeze(1)              
        s = s.mean(dim=1)                                               

        z = torch.cat([s, node_irregularity], dim=-1)                  
        attn_weights = F.softmax(self.fc2(self.act(self.fc1(z))), dim=-1)  

        V = (U_stack * attn_weights.view(B, len(self.kernels), 1, 1)).sum(dim=1)  

        proj = self.project(x) * presence_mask
        out = (V + proj) * presence_mask
        out = self.final_act(out) * presence_mask

        return out, attn_weights

class FixedTemporalInception(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=None):
        super().__init__()
        self.kernels = kernels if kernels is not None else [1, 3, 5, 7, 9, 11]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2, bias=True),
                nn.GroupNorm(1, out_channels),
                nn.GELU()
            ) for k in self.kernels
        ])
        self.project = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.final_act = nn.GELU()

    def forward(self, x, presence_mask, node_irregularity):
        B, C, T = x.shape
        branch_outs = []
        for conv in self.convs:
            out = conv(x)
            out = out * presence_mask
            branch_outs.append(out)

        U_stack = torch.stack(branch_outs, dim=1) 
        attn_weights = torch.full(
            (B, len(self.kernels)), 1.0 / len(self.kernels), device=x.device, dtype=x.dtype
        )

        V = (U_stack * attn_weights.view(B, len(self.kernels), 1, 1)).sum(dim=1)

        proj = self.project(x) * presence_mask
        out = (V + proj) * presence_mask
        out = self.final_act(out) * presence_mask

        return out, attn_weights

# ==========================================
# 3. 终极主模型: AEGIS
# ==========================================
class AEGIS(nn.Module):
    def __init__(self, node_in, edge_in, hidden, num_classes, seq_len=10, heads=8, dropout=0.3, 
                 drop_path=0.1, dropedge_p=0.2, kernels=None,
                 use_macro=True, use_micro=True, use_spatial_gating=True, use_edge_aug=True, temporal_mode="adaptive", 
                 **kwargs):
        super(AEGIS, self).__init__()
        
        # [防爆断言] 确保维度整除，防止隐藏 Bug
        assert hidden % 4 == 0, "hidden must be divisible by 4 for MacroTopologyGNN"
        assert hidden % heads == 0, "hidden must be divisible by heads for attention blocks"
        
        self.hidden, self.seq_len = hidden, seq_len
        self.dropedge_p = float(dropedge_p)
        
        self.use_macro = use_macro
        self.use_micro = use_micro
        self.use_spatial_gating = use_spatial_gating
        self.use_edge_aug = use_edge_aug
        self.temporal_mode = temporal_mode

        self.node_enc = nn.Sequential(nn.Linear(node_in, hidden), nn.LayerNorm(hidden))
        self.edge_enc = nn.Sequential(nn.Linear(edge_in, hidden), nn.LayerNorm(hidden))

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

        self.tpe = nn.Embedding(seq_len, hidden)
        self.edge_tpe = nn.Embedding(seq_len, hidden) 
        
        if self.temporal_mode == "adaptive":
            self.stream_temporal = AdaptiveTemporalInception(hidden, hidden, kernels=kernels)
        else:
            self.stream_temporal = FixedTemporalInception(hidden, hidden, kernels=kernels)

        self.aux_proj = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 3),
            nn.LayerNorm(hidden * 3),
            nn.GELU()
        )
        self.reconstruct_head = nn.Sequential(nn.Linear(hidden * 3, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden))
        
        self.edge_temporal = nn.GRU(
            input_size=hidden, hidden_size=hidden, num_layers=1, batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes)
        )

    # ================= 辅助对齐与记忆函数 =================
    def reset_mask_history(self):
        """兼容旧训练脚本的空操作；当前版本不再维护 Python 掩码历史。"""
        return None

    def _align_node_sequences(self, graphs, spatial_node_feats, raw_global_ids_seq):
        device = spatial_node_feats[0].device
        all_keys = []
        per_t_counts = []
        
        for t in range(self.seq_len):
            b_ids = graphs[t].batch if hasattr(graphs[t], "batch") and graphs[t].batch is not None else torch.zeros_like(raw_global_ids_seq[t])
            keys = torch.stack([b_ids, raw_global_ids_seq[t]], dim=1) 
            all_keys.append(keys)
            per_t_counts.append(keys.size(0))
            
        all_keys_tensor = torch.cat(all_keys, dim=0)
        unique_keys, inverse_indices = torch.unique(all_keys_tensor, dim=0, return_inverse=True)
        num_unique = unique_keys.size(0)

        dense_stack = torch.zeros((num_unique, self.seq_len, self.hidden), device=device)
        presence_mask = torch.zeros((num_unique, self.seq_len), device=device, dtype=torch.bool)
        
        offset = 0
        per_t_node_idx = []
        for t in range(self.seq_len):
            N_t = per_t_counts[t]
            if N_t > 0:
                idx_t = inverse_indices[offset : offset + N_t]
                dense_stack[idx_t, t, :] = spatial_node_feats[t]
                presence_mask[idx_t, t] = True
                per_t_node_idx.append(idx_t)
            else:
                per_t_node_idx.append(torch.empty(0, dtype=torch.long, device=device))
            offset += N_t

        node_batch_ids = unique_keys[:, 0]
        return dense_stack, presence_mask, node_batch_ids, per_t_node_idx

    def _align_target_edge_sequences(self, graphs, spatial_edge_feats, active_edge_indices):
        """仅为最后一帧需要分类的边回溯历史，避免对窗口内全部 unique edges 做全量对齐。"""
        device = spatial_edge_feats[0].device
        t_last = self.seq_len - 1

        last_src, _ = active_edge_indices[t_last]
        num_target_edges = last_src.size(0)
        if num_target_edges == 0:
            return (
                torch.zeros((0, self.seq_len, self.hidden), device=device),
                torch.zeros((0, self.seq_len), device=device, dtype=torch.bool),
            )

        if not hasattr(graphs[t_last], "edge_uid") or graphs[t_last].edge_uid is None:
            raise ValueError("Each frame must provide `edge_uid` for multigraph-safe edge alignment.")

        target_uids = graphs[t_last].edge_uid.long()
        if target_uids.size(0) != num_target_edges:
            raise ValueError("graphs[t_last].edge_uid must align with active_edge_indices[t_last].")

        if hasattr(graphs[t_last], "batch") and graphs[t_last].batch is not None:
            target_batch = graphs[t_last].batch[last_src].long()
        else:
            target_batch = torch.zeros(num_target_edges, dtype=torch.long, device=device)

        target_keys = torch.stack([target_batch, target_uids], dim=1)
        edge_dense = torch.zeros((num_target_edges, self.seq_len, self.hidden), device=device)
        edge_presence = torch.zeros((num_target_edges, self.seq_len), device=device, dtype=torch.bool)

        for t in range(self.seq_len):
            src_t, _ = active_edge_indices[t]
            E_t = src_t.size(0)
            if E_t == 0:
                continue

            if not hasattr(graphs[t], "edge_uid") or graphs[t].edge_uid is None:
                raise ValueError("Each frame must provide `edge_uid` for multigraph-safe edge alignment.")

            hist_uids = graphs[t].edge_uid.long()
            if hist_uids.size(0) != E_t:
                raise ValueError("graphs[t].edge_uid must align with active_edge_indices[t].")

            if hasattr(graphs[t], "batch") and graphs[t].batch is not None:
                hist_batch = graphs[t].batch[src_t].long()
            else:
                hist_batch = torch.zeros(E_t, dtype=torch.long, device=device)

            hist_keys = torch.stack([hist_batch, hist_uids], dim=1)
            all_keys = torch.cat([target_keys, hist_keys], dim=0)
            _, inv = torch.unique(all_keys, dim=0, return_inverse=True)

            inv_target = inv[:num_target_edges]
            inv_hist = inv[num_target_edges:]
            sorted_target, target_pos = torch.sort(inv_target)
            insert_pos = torch.searchsorted(sorted_target, inv_hist)

            valid = insert_pos < num_target_edges
            valid[valid] = sorted_target[insert_pos[valid]] == inv_hist[valid]
            if not valid.any():
                continue

            matched_hist_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            matched_target_idx = target_pos[insert_pos[valid]]

            edge_dense[matched_target_idx, t, :] = spatial_edge_feats[t][matched_hist_idx]
            edge_presence[matched_target_idx, t] = True

        t_idx = torch.arange(self.seq_len, device=device).unsqueeze(0)
        edge_tpe_feat = self.edge_tpe(t_idx)
        edge_dense = edge_dense + edge_tpe_feat * edge_presence.unsqueeze(-1).float()
        return edge_dense, edge_presence

    def _aggregate_graph_irregularity(self, batch_graph_irregs, num_graphs, device):
        padded = []
        mask = []
        for g in batch_graph_irregs:
            g_pad = torch.zeros(num_graphs, device=device)
            m_pad = torch.zeros(num_graphs, device=device)
            if g.size(0) > 0:
                g_pad[:g.size(0)] = g
                m_pad[:g.size(0)] = 1.0
            padded.append(g_pad)
            mask.append(m_pad)
            
        padded = torch.stack(padded, dim=0)
        mask = torch.stack(mask, dim=0)
        valid_counts = mask.sum(dim=0).clamp_min(1.0)
        return padded.sum(dim=0) / valid_counts

    def _run_edge_temporal(self, edge_dense, edge_presence):
        device = edge_dense.device
        E, T, H = edge_dense.shape

        if E == 0:
            return (torch.zeros((0, T, H), device=device), torch.zeros((0, H), device=device), torch.zeros((0,), device=device, dtype=torch.long))

        edge_lengths = edge_presence.sum(dim=1).clamp_min(1).long()  

        compact_dense = torch.zeros_like(edge_dense)
        for i in range(E):
            valid_t = torch.nonzero(edge_presence[i], as_tuple=False).squeeze(-1)
            if valid_t.numel() > 0:
                compact_dense[i, :valid_t.numel(), :] = edge_dense[i, valid_t, :]

        packed = pack_padded_sequence(compact_dense, lengths=edge_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.edge_temporal(packed)
        edge_temporal_all, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)  

        valid_mask = (torch.arange(T, device=device).unsqueeze(0) < edge_lengths.unsqueeze(1)) 
        edge_temporal_all = edge_temporal_all * valid_mask.unsqueeze(-1)
        edge_last_hidden = h_n[-1] 

        return edge_temporal_all, edge_last_hidden, edge_lengths

    # ================= 业务计算逻辑 =================
    def compute_degree_irregularity(self, edge_index, num_nodes):
        if edge_index.size(1) == 0:
            return (
                torch.zeros(num_nodes, device=edge_index.device),
                torch.zeros(0, device=edge_index.device)
            )

        src, dst = edge_index

        out_deg = degree(src, num_nodes, dtype=torch.float)
        in_deg  = degree(dst, num_nodes, dtype=torch.float)

        out_sum = out_deg.sum()
        in_sum = in_deg.sum()

        p_out = out_deg / (out_sum + 1e-8)
        p_in  = in_deg / (in_sum + 1e-8)

        node_irregularity = -0.5 * (torch.log(p_out + 1e-8) + torch.log(p_in + 1e-8))
        node_irregularity = node_irregularity / (node_irregularity.mean() + 1e-8)

        p_src_out = out_deg[src] / (out_sum + 1e-8)
        p_dst_in  = in_deg[dst] / (in_sum + 1e-8)
        edge_irregularity = -0.5 * (torch.log(p_src_out + 1e-8) + torch.log(p_dst_in + 1e-8))

        return node_irregularity, edge_irregularity

    def irregularity_guided_dropedge(self, edge_index, edge_attr, edge_irregularity, dropedge_p):
        if not self.training or float(dropedge_p) <= 0.0 or edge_index.size(1) == 0:
            return edge_index, edge_attr, torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)

        relative_irregularity = edge_irregularity / (edge_irregularity.mean() + 1e-6)
        drop_prob = float(dropedge_p) * (0.75 + 0.25 * torch.tanh(relative_irregularity - 1.0))
        drop_prob = torch.clamp(drop_prob, 0.0, min(0.5, float(dropedge_p) * 1.5))
        
        keep_prob = 1.0 - drop_prob
        edge_mask = torch.rand_like(keep_prob) < keep_prob

        return edge_index[:, edge_mask], edge_attr[edge_mask], edge_mask

    def compute_latent_denoising_loss(self, edge_rep, target_feat, global_irreg_scalar):
        """纯张量版 Masked Latent Edge Denoising，去掉 Python 历史字典与 CPU 同步。"""
        device = edge_rep.device
        E = target_feat.size(0)

        if E == 0:
            return torch.tensor(0.0, device=device)

        feat_dim = target_feat.size(1)
        mask_prob_tensor = torch.clamp(
            0.15 + 0.45 * torch.tanh(global_irreg_scalar),
            min=0.15,
            max=0.60,
        )

        keep_mask = (torch.rand(E, 1, device=device) > mask_prob_tensor).float()
        edge_feat_masked = target_feat * keep_mask
        edge_rep_corrupted = torch.cat([edge_feat_masked, edge_rep[:, feat_dim:]], dim=1)

        latent = self.aux_proj(edge_rep_corrupted)
        reconstructed = self.reconstruct_head(latent)

        per_edge_loss = F.mse_loss(
            reconstructed, target_feat.detach(), reduction='none'
        ).mean(dim=1)

        edge_weight = 1.0 + torch.tanh(global_irreg_scalar.squeeze(1))
        masked_edges = 1.0 - keep_mask.squeeze(1)
        denom = masked_edges.sum().clamp_min(1.0)

        return (per_edge_loss * edge_weight * masked_edges).sum() / denom

    def _spatial_encode_one_frame(self, data, dropedge_p):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, "batch") else None

        raw_global_ids = getattr(data, "global_node_id", getattr(data, "n_id", None))
        if raw_global_ids is None:
            raise ValueError("Each frame must provide `global_node_id` or `n_id` for temporal alignment.")

        if not hasattr(data, "edge_uid") or data.edge_uid is None:
            raise ValueError("Each frame must provide `edge_uid` for multigraph-safe edge alignment.")

        if data.edge_uid.size(0) != edge_index.size(1):
            raise ValueError("data.edge_uid must align with edge_index columns.")
        
        node_irreg, edge_irreg = self.compute_degree_irregularity(edge_index, x.size(0))

        if batch is not None and edge_index.size(1) > 0:
            edge_batch = batch[edge_index[0]]
            num_graphs = int(batch.max().item()) + 1
            graph_irreg_scalar = scatter(edge_irreg, edge_batch, dim=0, dim_size=num_graphs, reduce='mean')
        else:
            graph_irreg_scalar = edge_irreg.mean().view(1) if edge_index.size(1) > 0 else x.new_zeros(1)

        edge_index_d, edge_attr_d, edge_mask = self.irregularity_guided_dropedge(
            edge_index, edge_attr, edge_irreg, dropedge_p
        )

        x_base = self.node_enc(x)
        e_base = self.edge_enc(edge_attr)  
        e_base_d = e_base[edge_mask]       

        # Macro
        x_macro = x_base
        if self.use_macro:
            for layer in self.macro_spatial_layers:
                x_macro = layer(x_macro, edge_index_d, batch)
        else:
            x_macro = torch.zeros_like(x_base)

        # Micro
        x_micro, e_micro = x_base, e_base
        if self.use_micro:
            for layer in self.micro_spatial_layers:
                if self.use_edge_aug:
                    x_micro = layer["node_att"](x_micro, edge_index_d, e_base_d, batch)
                else:
                    x_micro = layer["node_att"](x_micro, edge_index_d, batch)
                e_micro = layer["edge_upd"](x_micro, edge_index, e_micro)
                e_base_d = e_micro[edge_mask]
        else:
            x_micro = torch.zeros_like(x_base)

        # Fusion
        if self.use_macro and self.use_micro:
            if self.use_spatial_gating:
                x_fused, _ = self.spatial_gating(x_macro, x_micro, node_irreg)
            else:
                x_fused = 0.5 * (x_macro + x_micro)
        else:
            x_fused = x_macro if self.use_macro else x_micro

        return x_fused, e_micro, edge_index, edge_mask, raw_global_ids, graph_irreg_scalar

    def forward(self, graphs):
        spatial_node_feats, spatial_edge_feats = [], []
        active_edge_indices, edge_masks = [], []
        raw_global_ids_seq, batch_graph_irregs = [], []

        device = graphs[0].x.device

        # -------------------------
        # Phase 1: Spatial
        # -------------------------
        for t in range(self.seq_len):
            x, edge_feat, edge_idx_act, e_mask, raw_ids, g_irreg = self._spatial_encode_one_frame(
                graphs[t], self.dropedge_p
            )
            raw_global_ids_seq.append(raw_ids)
            edge_masks.append(e_mask)
            active_edge_indices.append(edge_idx_act)  
            spatial_node_feats.append(x)
            spatial_edge_feats.append(edge_feat)
            batch_graph_irregs.append(g_irreg)

        # -------------------------
        # Phase 2: Node alignment
        # -------------------------
        dense_stack, presence_mask, node_batch_ids, per_t_node_idx = self._align_node_sequences(
            graphs, spatial_node_feats, raw_global_ids_seq
        )

        num_unique_nodes = dense_stack.size(0)
        if num_unique_nodes == 0:
            aux_loss = torch.tensor(0.0, device=device)
            logits = torch.zeros((0, self.classifier[-1].out_features), device=device)
            return logits, aux_loss

        t_emb = self.tpe(torch.arange(self.seq_len, device=device)).unsqueeze(0)   
        presence_feat = presence_mask.float().unsqueeze(-1)                         
        x_temporal_feat = dense_stack + t_emb * presence_feat

        if hasattr(graphs[0], "batch") and graphs[0].batch is not None:
            num_graphs = int(graphs[0].batch.max().item()) + 1
        else:
            num_graphs = 1

        mean_graph_irreg = self._aggregate_graph_irregularity(batch_graph_irregs, num_graphs, device)  
        node_graph_irreg = mean_graph_irreg[node_batch_ids].unsqueeze(-1)                               

        x_temporal_out, kernel_weights = self.stream_temporal(
            x_temporal_feat.permute(0, 2, 1),             
            presence_mask.unsqueeze(1).float(),           
            node_graph_irreg                              
        )
        dense_out = x_temporal_out.permute(0, 2, 1)       

        # -------------------------
        # Phase 3: Edge alignment + edge temporal encoder
        # -------------------------
        edge_dense, edge_presence = self._align_target_edge_sequences(
            graphs, spatial_edge_feats, active_edge_indices
        )

        if edge_dense.size(0) > 0:
            _, edge_last_hidden, _ = self._run_edge_temporal(edge_dense, edge_presence)
        else:
            edge_last_hidden = torch.zeros((0, self.hidden), device=device)

        # -------------------------
        # Phase 4: Last-frame readout
        # -------------------------
        t_last = self.seq_len - 1
        last_node_idx = per_t_node_idx[t_last]
        node_out_last = dense_out[last_node_idx, t_last, :]     

        src, dst = active_edge_indices[t_last][0], active_edge_indices[t_last][1]

        if src.numel() == 0:
            logits = torch.zeros((0, self.classifier[-1].out_features), device=device)
            aux_loss = torch.tensor(0.0, device=device)
            self._last_edge_masks = edge_masks
            self._last_kernel_weights = kernel_weights
            return logits, aux_loss

        edge_temporal_out = (
            edge_last_hidden
            if edge_last_hidden.size(0) > 0
            else spatial_edge_feats[t_last]
        )

        edge_rep = torch.cat([
            edge_temporal_out,
            node_out_last[src],
            node_out_last[dst]
        ], dim=1)

        logits = self.classifier(edge_rep)

        # -------------------------
        # Phase 5: Auxiliary loss
        # -------------------------
        aux_loss = torch.tensor(0.0, device=device)

        if self.training:
            aux_loss_sum = torch.tensor(0.0, device=device)
            valid_aux_steps = 0

            for t_aux in range(self.seq_len - 1):
                if spatial_edge_feats[t_aux].size(0) == 0:
                    continue

                node_idx_aux = per_t_node_idx[t_aux]
                node_out_aux = dense_out[node_idx_aux, t_aux, :]

                src_aux, dst_aux = active_edge_indices[t_aux][0], active_edge_indices[t_aux][1]

                edge_rep_aux = torch.cat([
                    spatial_edge_feats[t_aux],
                    node_out_aux[src_aux],
                    node_out_aux[dst_aux]
                ], dim=1)

                if hasattr(graphs[t_aux], "batch") and graphs[t_aux].batch is not None:
                    frame_irreg_scalar = batch_graph_irregs[t_aux][graphs[t_aux].batch[src_aux]]
                else:
                    frame_irreg_scalar = batch_graph_irregs[t_aux].expand(src_aux.size(0))

                step_loss = self.compute_latent_denoising_loss(
                    edge_rep=edge_rep_aux,
                    target_feat=spatial_edge_feats[t_aux],
                    global_irreg_scalar=frame_irreg_scalar.unsqueeze(1),
                )

                aux_loss_sum = aux_loss_sum + step_loss
                valid_aux_steps += 1

            if valid_aux_steps > 0:
                aux_loss = aux_loss_sum / valid_aux_steps

        self._last_edge_masks = edge_masks
        self._last_kernel_weights = kernel_weights

        return logits, aux_loss