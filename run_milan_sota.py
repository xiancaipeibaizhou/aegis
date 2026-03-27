import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import joblib
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from datetime import datetime
from tqdm import tqdm

# 导入超参解析和你的模型组
from hparams_a3 import resolve_hparams
from aegis import AEGIS
from ablation_models import (
    AEGIS_WoMacro, AEGIS_WoMicro, AEGIS_WoSpatialGating, 
    AEGIS_WoEdgeAug, AEGIS_FixedTemporal
)

# 默认开启 PyTorch 的 TF32 硬件加速 (4090 的默认神技，FP32精度+FP16速度)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ==========================================
# 1. 核心时序数据加载逻辑
# ==========================================
class TemporalGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_data_seq, seq_len=10, stride=1):
        self.graph_data_seq = [g for g in graph_data_seq if g is not None]
        self.seq_len = seq_len
        self.stride = max(1, int(stride))

    def __len__(self):
        n = len(self.graph_data_seq) - self.seq_len
        return 0 if n < 0 else (n // self.stride) + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        return self.graph_data_seq[start : start + self.seq_len]

def temporal_collate_fn(batch):
    if len(batch) == 0: return []
    seq_len = len(batch[0])
    batched_seq = []
    for t in range(seq_len):
        graphs_at_t = [sample[t] for sample in batch]
        batched_seq.append(Batch.from_data_list(graphs_at_t))
    return batched_seq

def get_normal_indices(class_names):
    if class_names is None: return [0]
    normals = []
    for idx, name in enumerate(class_names):
        name_lower = str(name).lower().replace('-', '').replace('_', '').replace(' ', '')
        if any(k in name_lower for k in ['benign', 'normal', 'nonvpn', 'nontor']):
            normals.append(idx)
    return normals if len(normals) > 0 else [0]

# ==========================================
# 2. 动态 F1 阈值与统一评估模块
# ==========================================
def find_best_macro_f1_threshold_and_predict(y_true_val, y_prob_val, y_prob_test, normal_indices):
    normal_probs_val = np.sum(y_prob_val[:, normal_indices], axis=1)
    attack_probs_val = 1.0 - normal_probs_val
    y_true_val_bin = (~np.isin(y_true_val, normal_indices)).astype(int)
    
    candidates = np.unique(np.quantile(attack_probs_val, np.linspace(0.0, 1.0, 101)))
    best_th, best_macro_f1, best_far = 0.5, -1.0, 1.0
    
    for th in candidates:
        y_pred_val_sim = np.argmax(y_prob_val, axis=-1)
        best_normal_class_val = np.array(normal_indices)[np.argmax(y_prob_val[:, normal_indices], axis=1)]
        y_pred_val_sim[attack_probs_val < th] = best_normal_class_val[attack_probs_val < th]
        
        mask = (attack_probs_val >= th) & np.isin(y_pred_val_sim, normal_indices)
        if mask.any():
            probs_copy = y_prob_val.copy()
            probs_copy[:, normal_indices] = -1.0 
            y_pred_val_sim[mask] = np.argmax(probs_copy[mask], axis=-1)
            
        y_pred_val_bin = (~np.isin(y_pred_val_sim, normal_indices)).astype(int)
        fp = np.logical_and(y_true_val_bin == 0, y_pred_val_bin == 1).sum()
        tn = np.logical_and(y_true_val_bin == 0, y_pred_val_bin == 0).sum()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        macro_f1 = f1_score(y_true_val, y_pred_val_sim, average='macro', zero_division=0)
        
        if macro_f1 > best_macro_f1 or (macro_f1 == best_macro_f1 and far < best_far):
            best_macro_f1, best_th, best_far = macro_f1, th, far

    test_preds = np.argmax(y_prob_test, axis=-1)
    normal_probs_test = np.sum(y_prob_test[:, normal_indices], axis=1)
    attack_probs_test = 1.0 - normal_probs_test
    
    best_normal_class_test = np.array(normal_indices)[np.argmax(y_prob_test[:, normal_indices], axis=1)]
    test_preds[attack_probs_test < best_th] = best_normal_class_test[attack_probs_test < best_th]
    
    mask_to_attack = (attack_probs_test >= best_th) & np.isin(test_preds, normal_indices)
    if mask_to_attack.any():
        probs_copy = y_prob_test.copy()
        probs_copy[:, normal_indices] = -1.0 
        test_preds[mask_to_attack] = np.argmax(probs_copy[mask_to_attack], axis=-1)
        
    return test_preds, best_th, best_macro_f1, best_far

def compute_all_metrics(y_true, y_pred, y_prob=None, class_names=None, normal_indices=None):
    metrics = {}
    metrics['ACC'] = accuracy_score(y_true, y_pred)
    metrics['APR'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['RE'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['F1 (Weighted)'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['F1 (Macro)'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    if y_prob is not None and y_prob.ndim == 2:
        present_classes = np.unique(y_true)
        if len(present_classes) < 2:
            metrics['AUC'] = float('nan')
        elif len(present_classes) == 2:
            pos_class = present_classes[1]
            y_true_bin = (y_true == pos_class).astype(int)
            try: metrics['AUC'] = roc_auc_score(y_true_bin, y_prob[:, pos_class])
            except: metrics['AUC'] = float('nan')
        else:
            aucs = []
            for c in present_classes:
                y_true_bin = (y_true == c).astype(int)
                if len(np.unique(y_true_bin)) == 2:
                    try: aucs.append(roc_auc_score(y_true_bin, y_prob[:, c]))
                    except: pass
            metrics['AUC'] = np.mean(aucs) if len(aucs) > 0 else float('nan')
    else: 
        metrics['AUC'] = float('nan')

    num_classes = len(class_names) if class_names is not None else int(np.max(y_true)) + 1
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    
    if normal_indices is None: normal_indices = [0]
    
    is_true_normal = np.isin(y_true, normal_indices)
    is_pred_normal = np.isin(y_pred, normal_indices)
    
    fp = np.logical_and(is_true_normal, ~is_pred_normal).sum()
    tn = np.logical_and(is_true_normal, is_pred_normal).sum()
    metrics['FAR'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    
    attack_mask = ~is_true_normal
    attack_total = attack_mask.sum()
    attack_correct = np.logical_and(attack_mask, y_true == y_pred).sum()
    metrics['ASA'] = float(attack_correct / attack_total) if attack_total > 0 else 0.0

    return metrics, cm

def plot_and_save_confusion_matrix(cm, target_names, save_path):
    clean_target_names = [str(name).replace('\x96', '-').replace('\u2013', '-') for name in target_names]
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) 
    
    num_classes = len(clean_target_names)
    fig_width = max(10, num_classes * 1.0)
    fig_height = max(8, num_classes * 0.8)
    
    plt.figure(figsize=(fig_width, fig_height))
    sns.set(font_scale=1.0) 
    
    annot = np.empty_like(cm_norm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{int(cm[i, j])}\n({cm_norm[i, j]*100:.1f}%)" if cm[i, j] > 0 else ""

    sns.heatmap(cm_norm, annot=annot, fmt="", cmap='Blues', cbar=True,
                xticklabels=clean_target_names, yticklabels=clean_target_names, vmin=0.0, vmax=1.0)
    
    plt.title('Normalized Confusion Matrix', pad=20, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 3. 评估获取 Logits
# ==========================================
@torch.no_grad()
def get_eval_predictions(model, loader, device, use_amp=False, amp_dtype=torch.float16):
    model.eval()
    all_labels, all_probs = [], []

    for batched_seq in loader:
        batched_seq = [g.to(device, non_blocking=True) for g in batched_seq]
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            out = model(batched_seq)
            logits, _ = out if isinstance(out, tuple) else (out, None)

        probs = torch.softmax(logits.float(), dim=-1)
        labels = batched_seq[-1].edge_labels

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_probs)

# ==========================================
# 4. 主流程 (纯净稳定版两阶段框架)
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='unsw_nb15')
    parser.add_argument('--data_dir', type=str, default='../processed_data')
    parser.add_argument('--variant', type=str, default='AEGIS',
                        choices=['AEGIS', 'WoMacro', 'WoMicro', 'WoSpatialGating', 'WoEdgeAug', 'FixedTemporal'])
    # 允许传入已预训练好的底座路径
    parser.add_argument('--pretrained_path', type=str, default='', help='Path to pre-trained backbone model')
    # 单独预训练开关
    parser.add_argument('--pretrain_only', action='store_true', help='Exit the script immediately after pre-training')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 先解析超参数，获取字典 h
    group_str = os.getenv("HP_GROUPS", "NB_EXP1_BASE").split(",")[0].strip()
    h = resolve_hparams(group_str, env=os.environ, dataset=args.dataset)
    
    # 2. 再从 h 中提取 seed 并固定随机种子
    import random
    seed = int(h.get("SEED", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    seq_len = int(h["SEQ_LEN"])
    batch_size = int(h["BATCH_SIZE"])
    num_epochs = int(h["NUM_EPOCHS"])
    lr = float(h["LR"])
    weight_decay = float(h.get("WEIGHT_DECAY", 1e-4)) 
    hidden = int(h["HIDDEN"])
    heads = int(h["HEADS"])
    kernels = list(h["KERNELS"])
    max_cl_edges = int(h.get("MAX_CL_EDGES", 8192))
    patience = int(h["PATIENCE"])
    accum_steps = max(1, int(h.get("ACCUM_STEPS", 1)))
    drop_path = float(h.get("DROP_PATH", 0.1))
    dropedge_p = float(h.get("DROPEDGE_P", 0.2))
    
    pretrain_epochs = int(h.get("PRETRAIN_EPOCHS", 10))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.variant}_{group_str}_dim{hidden}_seq{seq_len}"
    save_dir = os.path.join("results", args.dataset, exp_name, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 Outputs will be saved to: {save_dir}")

    dataset_path = os.path.join(args.data_dir, args.dataset)
    if not os.path.exists(dataset_path):
        fallback_dataset_path = os.path.join("processed_data", args.dataset)
        if os.path.exists(fallback_dataset_path):
            dataset_path = fallback_dataset_path
        else:
            raise FileNotFoundError(
                f"Dataset directory not found: {dataset_path} or {fallback_dataset_path}"
            )

    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    use_grad_scaler = use_amp and amp_dtype == torch.float16
    scaler_pt = torch.amp.GradScaler('cuda', enabled=use_grad_scaler)
    scaler_ft = torch.amp.GradScaler('cuda', enabled=use_grad_scaler)

    train_graphs = torch.load(os.path.join(dataset_path, "train_graphs.pt"), weights_only=False)
    val_graphs = torch.load(os.path.join(dataset_path, "val_graphs.pt"), weights_only=False)
    test_graphs = torch.load(os.path.join(dataset_path, "test_graphs.pt"), weights_only=False)

    label_enc_path = os.path.join(dataset_path, "label_encoder.pkl")
    if os.path.exists(label_enc_path):
        class_names = joblib.load(label_enc_path).classes_
        num_classes = len(class_names)
    else:
        max_label = 0
        for split_graphs in [train_graphs, val_graphs, test_graphs]:
            for g in split_graphs:
                if g is not None and g.edge_labels.numel() > 0:
                    max_label = max(max_label, int(g.edge_labels.max().item()))
        num_classes = max_label + 1
        class_names = [f"Class_{i}" for i in range(num_classes)]

    counts = np.zeros(num_classes, dtype=np.float64)
    for g in train_graphs:
        if g is not None and g.edge_labels.numel() > 0:
            counts += np.bincount(g.edge_labels.cpu().numpy(), minlength=num_classes)

    weights_cpu = 1.0 / (np.sqrt(counts) + 1.0)
    weights_cpu = torch.tensor(weights_cpu / weights_cpu.sum() * num_classes, dtype=torch.float32)
    node_dim, edge_dim = train_graphs[0].x.shape[1], train_graphs[0].edge_attr.shape[1]

    # 🚀 提速优化: DataLoader 多进程并行预加载 + 锁页内存 (抛弃了 AMP，靠 IO 提速)
    train_stride = int(h.get("TRAIN_STRIDE", 2))

    train_loader = DataLoader(
        TemporalGraphDataset(train_graphs, seq_len, stride=train_stride),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=temporal_collate_fn,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    val_loader = DataLoader(
        TemporalGraphDataset(val_graphs, seq_len, stride=1),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=temporal_collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        TemporalGraphDataset(test_graphs, seq_len, stride=1),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=temporal_collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )


    model_kwargs = {
        "node_in": node_dim, "edge_in": edge_dim, "hidden": hidden, "num_classes": num_classes,
        "seq_len": seq_len, "heads": heads, "dropout": 0.3, "max_cl_edges": max_cl_edges,
        "kernels": kernels, "drop_path": drop_path, "dropedge_p": dropedge_p,
    }
    
    print(f"🚀 Initializing Model Variant: {args.variant}")
    if args.variant == "AEGIS": model = AEGIS(**model_kwargs).to(device)
    elif args.variant == "WoMacro": model = AEGIS_WoMacro(**model_kwargs).to(device)
    elif args.variant == "WoMicro": model = AEGIS_WoMicro(**model_kwargs).to(device)
    elif args.variant == "WoSpatialGating": model = AEGIS_WoSpatialGating(**model_kwargs).to(device)
    elif args.variant == "WoEdgeAug": model = AEGIS_WoEdgeAug(**model_kwargs).to(device)
    elif args.variant == "FixedTemporal": model = AEGIS_FixedTemporal(**model_kwargs).to(device)

    # ==========================================
    # 🔥 Phase 1: Generative Subgraph Pre-training (无监督)
    # ==========================================
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"\n🔄 [Stage 1] Skipping Pre-training. Loading pre-trained backbone from: {args.pretrained_path}")
        model.load_state_dict(torch.load(args.pretrained_path, map_location=device), strict=False)
        print("✅ Pre-trained Backbone successfully loaded!")

    elif pretrain_epochs > 0:
        print(f"\n🧠 [Stage 1] Self-Supervised Pre-training for {pretrain_epochs} Epochs...")
        
        optimizer_pt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler_pt = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_pt, T_0=10, T_mult=1, eta_min=lr*0.01)
        
        for epoch in range(pretrain_epochs):
            if hasattr(model, "reset_mask_history"):
                model.reset_mask_history()
            model.train()
            total_cl_loss, cl_steps = 0.0, 0
            optimizer_pt.zero_grad(set_to_none=True)
            
            for step, batched_seq in enumerate(tqdm(train_loader, desc=f"PT Epoch {epoch+1}", leave=False)):
                batched_seq = [g.to(device, non_blocking=True) for g in batched_seq]

                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    out = model(batched_seq)
                    _, aux_loss = out if isinstance(out, tuple) else (out, None)

                if torch.is_tensor(aux_loss) and aux_loss.requires_grad:
                    loss = aux_loss / float(accum_steps)
                    if use_grad_scaler:
                        scaler_pt.scale(loss).backward()
                    else:
                        loss.backward()
                    total_cl_loss += aux_loss.item()
                    cl_steps += 1

                if ((step + 1) % accum_steps == 0) or ((step + 1) == len(train_loader)):
                    if use_grad_scaler:
                        scaler_pt.unscale_(optimizer_pt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    if use_grad_scaler:
                        scaler_pt.step(optimizer_pt)
                        scaler_pt.update()
                    else:
                        optimizer_pt.step()
                    optimizer_pt.zero_grad(set_to_none=True)
                    
            scheduler_pt.step()
            avg_cl = total_cl_loss / max(1, cl_steps)
            print(f"PT Epoch {epoch+1:03d} | Reconstruction Loss: {avg_cl:.4f}")
            
        pt_save_path = os.path.join(save_dir, "pretrained_backbone.pth")
        torch.save(model.state_dict(), pt_save_path)
        print(f"✅ Pre-training Complete! Backbone saved to: {pt_save_path}")
        
        # 接收到 --pretrain_only 标志，直接光荣下岗
        if args.pretrain_only:
            print("🛑 [--pretrain_only] flag detected. Exiting script before fine-tuning.")
            return

    # ==========================================
    # 🔥 Phase 2: Supervised Fine-tuning (有监督微调)
    # ==========================================
    print(f"\n🎯 [Stage 2] Joint Supervised Fine-tuning for {num_epochs} Epochs...")
            
    optimizer_ft = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler_ft = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ft, T_0=h.get("COSINE_T0", 10), T_mult=1, eta_min=lr*0.01)
    criterion = nn.CrossEntropyLoss(weight=weights_cpu.to(device))

    # 获取对比学习权重
    cl_weight = float(h.get("CL_LOSS_WEIGHT", 0.5))
    print(f"⚖️ Loss Formulation: Total Loss = CE Loss + {cl_weight} * CL Loss")

    best_val_auprc = -1.0
    best_val_f1 = -1.0
    patience_cnt = 0
    training_log = []

    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
    
    for epoch in range(num_epochs):
        if hasattr(model, "reset_mask_history"):
            model.reset_mask_history()
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_cl = 0.0
        cl_active_steps = 0
        
        optimizer_ft.zero_grad(set_to_none=True)
        
        for step, batched_seq in enumerate(tqdm(train_loader, desc=f"FT Epoch {epoch+1}", leave=False)):
            batched_seq = [g.to(device, non_blocking=True) for g in batched_seq]

            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                out = model(batched_seq)
                logits, aux_loss = out if isinstance(out, tuple) else (out, None)

                last_frame_labels = batched_seq[-1].edge_labels
                ce_loss = criterion(logits.float(), last_frame_labels)

                batch_total_loss = ce_loss
                if torch.is_tensor(aux_loss) and aux_loss.requires_grad:
                    batch_total_loss = batch_total_loss + cl_weight * aux_loss

            if torch.is_tensor(aux_loss) and aux_loss.requires_grad:
                total_cl += aux_loss.item()
                cl_active_steps += 1

            loss = batch_total_loss / float(accum_steps)
            if use_grad_scaler:
                scaler_ft.scale(loss).backward()
            else:
                loss.backward()

            total_loss += batch_total_loss.item()
            total_ce += ce_loss.item()

            if ((step + 1) % accum_steps == 0) or ((step + 1) == len(train_loader)):
                if use_grad_scaler:
                    scaler_ft.unscale_(optimizer_ft)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                if use_grad_scaler:
                    scaler_ft.step(optimizer_ft)
                    scaler_ft.update()
                else:
                    optimizer_ft.step()
                optimizer_ft.zero_grad(set_to_none=True)

        scheduler_ft.step()
        
        # ================= SOTA 验证评估 =================
        val_true, val_probs = get_eval_predictions(
            model, val_loader, device, use_amp=use_amp, amp_dtype=amp_dtype
        )
        classes = np.arange(num_classes)
        val_true_bin = label_binarize(val_true, classes=classes)
        
        if num_classes == 2:
            val_true_bin = np.hstack((1 - val_true_bin, val_true_bin))
            
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 屏蔽验证集没有正样本时的烦人警告
            try:
                val_auprc_macro = average_precision_score(val_true_bin, val_probs, average='macro')
                # 如果依然算出 NaN (全是正常流量)，则安全置为 0.0，防止破坏早停判断
                if np.isnan(val_auprc_macro):
                    val_auprc_macro = 0.0
            except Exception:
                val_auprc_macro = 0.0
                
        val_preds_raw = np.argmax(val_probs, axis=-1)
        val_f1_macro = f1_score(val_true, val_preds_raw, average='macro', zero_division=0)
        
        # 修改这里以适配你 plot_training_logs.py 里的正则提取
        avg_ce = total_ce / max(1, len(train_loader))
        avg_cl = total_cl / max(1, cl_active_steps) if cl_active_steps > 0 else 0.0
        
        log_line = f"Epoch {epoch+1:03d} | Loss: {avg_ce:.4f} | CL: {avg_cl:.4f} | Val AUPRC: {val_auprc_macro:.4f} | Val F1: {val_f1_macro:.4f}"
        print(f"FT {log_line}")
        training_log.append(log_line)
        
        # 🌟 动态耐心值机制：如果 AUPRC 已经接近完美（>0.999），大幅缩短耐心防止过拟合
        current_patience = patience
        if val_auprc_macro >= 0.9990:
            current_patience = 5  # 学霸通道：满分后最多只多看 5 轮，不涨直接交卷

        is_auprc_improved = val_auprc_macro > best_val_auprc + 1e-4
        is_auprc_equal = abs(val_auprc_macro - best_val_auprc) <= 1e-4
        is_f1_improved = val_f1_macro > best_val_f1

        if is_auprc_improved or (is_auprc_equal and is_f1_improved):
            best_val_auprc = val_auprc_macro
            best_val_f1 = val_f1_macro
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            # 增加一行提示，让你清楚知道是哪个指标触发了保存
            trigger_reason = "AUPRC Improved" if is_auprc_improved else "AUPRC Equal, F1 Improved"
            print(f"  💾 [Model Saved] {trigger_reason} (AUPRC: {best_val_auprc:.4f}, F1: {best_val_f1:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= current_patience: 
                print(f"  🛑 Early stopping triggered at Epoch {epoch+1} (No improvement for {current_patience} epochs).")
                break

    # ====================
    # 测试与动态阈值应用
    # ====================
    print("\n[Testing] Evaluating Best Model...")
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth"), map_location=device))
    
    val_true, val_prob = get_eval_predictions(
        model, val_loader, device, use_amp=use_amp, amp_dtype=amp_dtype
    )
    test_true, test_prob = get_eval_predictions(
        model, test_loader, device, use_amp=use_amp, amp_dtype=amp_dtype
    )
    
    normal_indices = get_normal_indices(class_names)
        
    test_pred, best_th, val_macro, val_far = find_best_macro_f1_threshold_and_predict(val_true, val_prob, test_prob, normal_indices)
    metrics, cm = compute_all_metrics(test_true, test_pred, test_prob, class_names, normal_indices)
    plot_and_save_confusion_matrix(cm, class_names, os.path.join(save_dir, f"cm_thresh_{best_th:.2f}.png"))
    
    # ====================
    # 结果写入与保存
    # ====================
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"=== {exp_name} (Thresh: {best_th:.2f}) ===\n")
        for k, v in metrics.items(): 
            f.write(f"{k}: {v:.4f}\n")
            
    with open(os.path.join(save_dir, "training_history.log"), "w") as f:
        for log_line in training_log:
            f.write(log_line + "\n")
            
    csv_file = "aegis_results.csv" 
    if not os.path.isfile(csv_file):
        with open(csv_file, "w") as f:
            f.write("Dataset,Variant,Group,Threshold,ACC,APR,RE,F1_Macro,F1_Weighted,AUC,ASA,FAR\n")
            
    with open(csv_file, "a") as f:
        f.write(f"{args.dataset},{args.variant},{group_str},{best_th:.4f},"
                f"{metrics['ACC']:.4f},{metrics['APR']:.4f},{metrics['RE']:.4f},"
                f"{metrics['F1 (Macro)']:.4f},{metrics['F1 (Weighted)']:.4f},{metrics['AUC']:.4f},"
                f"{metrics['ASA']:.4f},{metrics['FAR']:.4f}\n")

if __name__ == "__main__":
    main()