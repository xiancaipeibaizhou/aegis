#!/bin/bash
# ==============================================================================
# 🌟 MILAN - Full Matrix SOTA Experiments
# 包含所有数据集的完整消融实验与超参敏感性分析
# 说明: 架构消融将自动读取 hparams_a3.py 中的 DATASET_BEST 最优基准配置
# ==============================================================================

# 定义要测试的数据集列表
# DATASETS=("unsw_nb15" "cic_ids2017" "iscx_ids2012" "darknet2020_block")

# 如果你想单独跑某一个，可以取消下面这行的注释并修改


cd MILAN
DATASETS=("darknet2020_block")
for DATASET in "${DATASETS[@]}"; do
    echo "================================================================="
    echo "🚀🚀🚀 STARTING FULL MATRIX EXPERIMENTS FOR: $DATASET 🚀🚀🚀"
    echo "================================================================="

    # ---------------------------------------------------------
    # 1. 核心架构消融 (Ablation Studies)
    # 策略: 使用 HP_GROUPS=DEFAULT，让模型自动加载针对该数据集的 SOTA 锚点
    # ---------------------------------------------------------
    echo "▶️ [1/6] Running Architecture Ablations (Using DATASET_BEST parameters)..."
    # HP_GROUPS=DEFAULT python run_milan_sota.py --dataset $DATASET --variant StandardTransformer
    # HP_GROUPS=DEFAULT python run_milan_sota.py --dataset $DATASET --variant WoLocal
    # HP_GROUPS=DEFAULT python run_milan_sota.py --dataset $DATASET --variant WoGlobal
    # HP_GROUPS=DEFAULT python run_milan_sota.py --dataset $DATASET --variant WoGating
    # HP_GROUPS=DEFAULT python run_milan_sota.py --dataset $DATASET --variant WoEdgeAug
    HP_GROUPS=DEFAULT python run_milan_sota.py --dataset $DATASET --variant MILAN

    # ---------------------------------------------------------
    # 2. 时序序列长度敏感性分析 (Sequence Length)
    # ---------------------------------------------------------
    # echo "▶️ [2/6] Running Sequence Length Search..."
    # HP_GROUPS=EXP_SEQ_2  python run_milan_sota.py --dataset $DATASET --variant MILAN
    # HP_GROUPS=EXP_SEQ_3  python run_milan_sota.py --dataset $DATASET --variant MILAN
    # HP_GROUPS=EXP_SEQ_5  python run_milan_sota.py --dataset $DATASET --variant MILAN
    # HP_GROUPS=EXP_SEQ_10 python run_milan_sota.py --dataset $DATASET --variant MILAN
    # HP_GROUPS=EXP_SEQ_20 python run_milan_sota.py --dataset $DATASET --variant MILAN

    # ---------------------------------------------------------
    # 3. 多尺度图核感受野 (Inception Kernels)
    # ---------------------------------------------------------
    # echo "▶️ [3/6] Running Inception Kernel Search..."
    # HP_GROUPS=EXP_KER_SINGLE python run_milan_sota.py --dataset $DATASET --variant MILAN
    # HP_GROUPS=EXP_KER_SHORT  python run_milan_sota.py --dataset $DATASET --variant MILAN
    # HP_GROUPS=EXP_KER_BASE   python run_milan_sota.py --dataset $DATASET --variant MILAN
    # HP_GROUPS=EXP_KER_LONG   python run_milan_sota.py --dataset $DATASET --variant MILAN

    # ---------------------------------------------------------
    # 4. 图结构鲁棒性分析 (DropEdge)
    # ---------------------------------------------------------
    # echo "▶️ [4/6] Running DropEdge Robustness Search..."
    # HP_GROUPS=EXP_DROP_0 python run_milan_sota.py --dataset $DATASET --variant MILAN
    # HP_GROUPS=EXP_DROP_1 python run_milan_sota.py --dataset $DATASET --variant MILAN
    # HP_GROUPS=EXP_DROP_2 python run_milan_sota.py --dataset $DATASET --variant MILAN
    # HP_GROUPS=EXP_DROP_4 python run_milan_sota.py --dataset $DATASET --variant MILAN

    # # ---------------------------------------------------------
    # # 5. 隐空间对比学习强度 (Contrastive Loss Weight)
    # # ---------------------------------------------------------
    # echo "▶️ [5/6] Running Contrastive Loss Weight Search..."
    # HP_GROUPS=EXP_CL_0  python run_milan_sota.py --dataset $DATASET --variant MILAN
    # HP_GROUPS=EXP_CL_02 python run_milan_sota.py --dataset $DATASET --variant MILAN
    # HP_GROUPS=EXP_CL_05 python run_milan_sota.py --dataset $DATASET --variant MILAN
    # HP_GROUPS=EXP_CL_08 python run_milan_sota.py --dataset $DATASET --variant MILAN
    # HP_GROUPS=EXP_CL_10 python run_milan_sota.py --dataset $DATASET --variant MILAN

    # # ---------------------------------------------------------
    # # 6. 模型容量与表征维度 (Hidden Capacity)
    # # ---------------------------------------------------------
    # echo "▶️ [6/6] Running Hidden Capacity Search..."
    # HP_GROUPS=EXP_CAP_TINY  python run_milan_sota.py --dataset $DATASET --variant MILAN
    # HP_GROUPS=EXP_CAP_SMALL python run_milan_sota.py --dataset $DATASET --variant MILAN
    # HP_GROUPS=EXP_CAP_BASE  python run_milan_sota.py --dataset $DATASET --variant MILAN
    # HP_GROUPS=EXP_CAP_LARGE python run_milan_sota.py --dataset $DATASET --variant MILAN

    echo "✅ Finished all experiments for $DATASET!"
    echo "-----------------------------------------------------------------"

done

echo "🎉 All datasets and full matrix experiments completed successfully!"


###########################333
#!/bin/bash
# ==============================================================================
# 🌟 MILAN - Full Matrix SOTA Experiments
# 包含所有数据集的完整消融实验与超参敏感性分析
# 说明: 架构消融将自动读取 hparams_a3.py 中的 DATASET_BEST 最优基准配置
# ==============================================================================

# 定义要测试的数据集列表
# DATASETS=("unsw_nb15" "cic_ids2017" "iscx_ids2012" "darknet2020_block")

# 如果你想单独跑某一个，可以取消下面这行的注释并修改


BATCH_SIZE=32 PRETRAIN_EPOCHS=30 python run_milan_sota.py --dataset darknet2020_block --variant AEGIS --pretrain_only
BATCH_SIZE=16 PRETRAIN_EPOCHS=30 python run_milan_sota.py --dataset unsw_nb15 --variant AEGIS --pretrain_only

BATCH_SIZE=64 NUM_EPOCHS=150 python run_milan_sota.py --dataset darknet2020_block --variant AEGIS --pretrained_path results/darknet2020_block/AEGIS_NB_EXP1_BASE_dim128_seq10/20260320-154721/pretrained_backbone.pth

BATCH_SIZE=32 NUM_EPOCHS=150 python run_milan_sota.py --dataset unsw_nb15 --variant AEGIS --pretrained_path results/unsw_nb15/AEGIS_NB_EXP1_BASE_dim128_seq20/20260320-155606/pretrained_backbone.pth

