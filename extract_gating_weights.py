import os
import re
import argparse
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

# 导入现有的超参和模型组件
from hparams_a3 import resolve_hparams
from aegis import AEGIS
from ablation_models import (
    AEGIS_WoMacro, AEGIS_WoMicro, AEGIS_WoSpatialGating, 
    AEGIS_WoEdgeAug, AEGIS_FixedTemporal
)

# ==========================================
# 1. 数据加载与拼接逻辑
# ==========================================
class TemporalGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_data_seq, seq_len=10):
        self.graph_data_seq = [g for g in graph_data_seq if g is not None]
        self.seq_len = seq_len
    def __len__(self):
        return max(0, len(self.graph_data_seq) - self.seq_len + 1)
    def __getitem__(self, idx):
        return self.graph_data_seq[idx : idx + self.seq_len]

def temporal_collate_fn(batch):
    if len(batch) == 0: return []
    seq_len = len(batch[0])
    batched_seq = []
    for t in range(seq_len):
        graphs_at_t = [sample[t] for sample in batch]
        batched_seq.append(Batch.from_data_list(graphs_at_t))
    return batched_seq

# ==========================================
# 2. 图表绘制函数 
# ==========================================
def plot_gating_distribution(df, exp_name, save_dir):
    # 清理类名中的乱码
    df['Attack Class'] = df['Attack Class'].apply(lambda x: str(x).replace('\x96', '-'))
    
    # 为了展示规律，按照宏观图结构熵从大到小排序 (体现熵越大，越偏向小卷积核)
    df = df.sort_values(by="Mean Graph Entropy", ascending=False)

    fig, ax1 = plt.subplots(figsize=(14, 7))
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    
    bar_width = 0.6
    x = np.arange(len(df))
    
    # 定义从冷色到暖色的渐变，代表从小感受野 (k=1) 到大感受野 (k=11)
    colors = ['#FF9999', '#FFCC99', '#FFFF99', '#CCFF99', '#99CCFF', '#CC99FF']
    
    # 动态获取 Kernel 列名
    kernel_cols = [c for c in df.columns if 'Kernel_' in c and '_Weight' in c]
    
    bottoms = np.zeros(len(df))
    for i, col in enumerate(kernel_cols):
        # 提取标题，例如 Kernel_1_Weight -> K=1
        label_name = col.replace('Kernel_', 'K=').replace('_Weight', '')
        ax1.bar(x, df[col], bar_width, bottom=bottoms, color=colors[i % len(colors)], label=label_name)
        bottoms += df[col]
    
    ax1.set_ylabel('Adaptive Receptive Field Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Traffic Type', fontsize=12, fontweight='bold')
    ax1.set_title('AEGIS: Adaptive Kernel Selection & Structural Entropy by Attack Type', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Attack Class'], rotation=45, ha='right')
    ax1.set_ylim(0, 1.0)
    
    # ✨ 在右侧 Y 轴绘制结构熵折线
    ax2 = ax1.twinx()
    p_entropy = ax2.plot(x, df['Mean Graph Entropy'], color='#2F5597', marker='s', markersize=8, linewidth=2.5, label='Mean Graph Entropy')
    ax2.set_ylabel('Mean Graph Entropy', color='#2F5597', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#2F5597')
    
    # 合并图例并将其放在图表右侧，避免遮挡数据
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='center left', bbox_to_anchor=(1.08, 0.5), framealpha=0.9, title="Modules")
    
    fig.tight_layout()
    save_path = os.path.join(save_dir, f"kernel_distribution_{exp_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 AEGIS 多尺度感受野权重分布图已保存至: {save_path}")

# ==========================================
# 3. 核心提取逻辑
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Extract and analyze gating weights & entropy from MILAN models.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the best_model.pth file")
    parser.add_argument('--data_dir', type=str, default='../processed_data', help="Directory of processed datasets")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件 {model_path}")
        return

    print(f"🔍 解析模型路径: {model_path}")
    parts = model_path.split(os.sep)
    dataset_name = parts[-4]
    exp_name = parts[-3]
    model_dir = os.path.dirname(model_path)
    
    match = re.match(r"^(.*?)_(.*)_dim(\d+)_seq(\d+)$", exp_name)
    if not match:
        print(f"❌ 错误: 无法解析实验参数: {exp_name}")
        return
        
    variant, group_str, hidden, seq_len = match.group(1), match.group(2), int(match.group(3)), int(match.group(4))
    
    dataset_path = os.path.join(args.data_dir, dataset_name)
    test_graphs = torch.load(os.path.join(dataset_path, "test_graphs.pt"), weights_only=False)

    label_enc_path = os.path.join(dataset_path, "label_encoder.pkl")
    if os.path.exists(label_enc_path):
        class_names = joblib.load(label_enc_path).classes_
    else:
        counts = np.zeros(100)
        for g in test_graphs: counts += np.bincount(g.edge_labels.numpy(), minlength=100)
        num_classes = int(np.max(np.nonzero(counts))) + 1
        class_names = [f"Class_{i}" for i in range(num_classes)]
        
    node_dim, edge_dim = test_graphs[0].x.shape[1], test_graphs[0].edge_attr.shape[1]
    test_loader = DataLoader(TemporalGraphDataset(test_graphs, seq_len), batch_size=32, shuffle=False, collate_fn=temporal_collate_fn)

    h = resolve_hparams(group_str, env=os.environ, dataset=dataset_name)
    model_kwargs = {
        "node_in": node_dim, "edge_in": edge_dim, "hidden": hidden, "num_classes": len(class_names),
        "seq_len": seq_len, "heads": int(h["HEADS"]), "dropout": 0.3, "max_cl_edges": int(h.get("MAX_CL_EDGES", 8192)),
        "kernels": list(h["KERNELS"]), "drop_path": float(h.get("DROP_PATH", 0.1)), "dropedge_p": float(h.get("DROPEDGE_P", 0.2)),
    }
    
    if variant == "AEGIS": model = AEGIS(**model_kwargs).to(device)
    elif variant == "WoMacro": model = AEGIS_WoMacro(**model_kwargs).to(device)
    elif variant == "WoMicro": model = AEGIS_WoMicro(**model_kwargs).to(device)
    elif variant == "WoSpatialGating": model = AEGIS_WoSpatialGating(**model_kwargs).to(device)
    elif variant == "WoEdgeAug": model = AEGIS_WoEdgeAug(**model_kwargs).to(device)
    elif variant == "FixedTemporal": model = AEGIS_FixedTemporal(**model_kwargs).to(device)
    else: 
        print(f"❌ 错误: 未知的变体 {variant}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if not hasattr(model, 'stream_temporal'):
        print(f"⚠️ 警告: 当前加载的模型 ({variant}) 不包含 stream_temporal 模块。")
        return

    extracted_kernel_weights = []
    extracted_entropies = []
    
    # ✨ 专为 AEGIS 定制的 Hook：提取时序核分布权重和宏观结构熵
    def temporal_hook(module, input_args, output):
        # output 是 (out, attn_weights)
        if isinstance(output, tuple) and len(output) > 1:
            extracted_kernel_weights.append(output[1].detach())
            # input_args[1] 是传入的 mean_graph_entropy
            if len(input_args) > 1:
                extracted_entropies.append(input_args[1].detach())
                
    model.stream_temporal.register_forward_hook(temporal_hook)

    print("🚀 开始推理并提取多尺度卷积核权重及熵值...")
    all_kernel_weights = []
    all_edge_entropies = []
    all_edge_labels = []

    with torch.no_grad():
        for batched_seq in tqdm(test_loader, desc="Evaluating"):
            extracted_kernel_weights.clear()
            extracted_entropies.clear()
            batched_seq_dev = [g.to(device) for g in batched_seq]
            
            _ = model(batched_seq_dev)
            
            if not extracted_kernel_weights or not extracted_entropies:
                continue
                
            # [Num_Unique_Nodes, Num_Kernels]
            batch_kernel_weights = extracted_kernel_weights[0] 
            entropy_val = extracted_entropies[0][0, 0].item() # 获取该批次的标量熵
            
            batch_global_ids = []
            for data in batched_seq_dev:
                if hasattr(data, "n_id"): batch_global_ids.append(data.n_id)
                elif hasattr(data, "id"): batch_global_ids.append(data.id)
                else: batch_global_ids.append(torch.arange(data.x.size(0), device=device))
                    
            all_ids = torch.cat(batch_global_ids)
            unique_ids, _ = torch.sort(torch.unique(all_ids))
            
            last_frame = batched_seq_dev[-1]
            indices = torch.searchsorted(unique_ids, batch_global_ids[-1])
            frame_node_weights = batch_kernel_weights[indices]
            
            src, dst = last_frame.edge_index[0], last_frame.edge_index[1]
            # 取边两端节点对于不同感受野权重的均值
            edge_weights = (frame_node_weights[src] + frame_node_weights[dst]) / 2.0
            
            all_kernel_weights.append(edge_weights.cpu().numpy())
            all_edge_entropies.append(np.full(edge_weights.shape[0], entropy_val))
            all_edge_labels.append(last_frame.edge_labels.cpu().numpy())

    all_kernel_weights = np.concatenate(all_kernel_weights, axis=0)
    all_edge_entropies = np.concatenate(all_edge_entropies)
    all_edge_labels = np.concatenate(all_edge_labels)
    
    # 获取模型里定义的 kernels 列表
    kernels = model.stream_temporal.kernels 
    
    results = []
    for class_idx, class_name in enumerate(class_names):
        mask = (all_edge_labels == class_idx)
        if mask.sum() > 0:
            avg_kernel_dist = np.mean(all_kernel_weights[mask], axis=0) # [Num_Kernels]
            avg_entropy = np.mean(all_edge_entropies[mask])
            
            res_dict = {
                "Attack Class": class_name,
                "Sample Count": mask.sum(),
                "Mean Graph Entropy": avg_entropy
            }
            # 动态填充每个 Kernel 的权重
            for k_idx, k_size in enumerate(kernels):
                res_dict[f"Kernel_{k_size}_Weight"] = avg_kernel_dist[k_idx]
                
            results.append(res_dict)
            
    df_results = pd.DataFrame(results)
    print("\n✅ 提取完成！各类别多尺度卷积核权重分布及平均结构熵：")
    print(df_results.to_string(index=False))

    csv_save_path = os.path.join(model_dir, f"kernel_weights_{exp_name}.csv")
    df_results.to_csv(csv_save_path, index=False)
    plot_gating_distribution(df_results, exp_name, model_dir)

if __name__ == "__main__":
    main()

# python extract_gating_weights.py --model_path results/darknet2020_block/AEGIS_NB_EXP1_BASE_dim128_seq10/20260320-165221/best_model.pth
# MILAN/results/cic_ids2017/MILAN_DEFAULT_dim128_seq3/20260317-233003/best_model.pth
# results/darknet2020_block/MILAN_DEFAULT_dim128_seq30/20260318-105715/best_model.pth