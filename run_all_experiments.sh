#!/bin/bash

# 1. 定义你要跑的 4 个数据集 (请根据你的实际文件夹名称修改)
DATASETS=("unsw_nb15")
# "unsw_nb15" "cic_ids2017" "iscx_ids2012" "darknet2020_block"
# 2. 定义你要跑的完整模型与消融变体
VARIANTS=("AEGIS")

# 3. 环境变量设置 (根据显存大小调整 BATCH_SIZE)
export BATCH_SIZE=128
export PRETRAIN_EPOCHS=30
export NUM_EPOCHS=150
# 创建一个总日志文件夹
mkdir -p global_logs

echo "🚀 开始全量实验自动化批量运行..."

# 4. 双重循环遍历执行
for variant in "${VARIANTS[@]}"; do
    for dataset in "${DATASETS[@]}"; do        

        LOG_FILE="global_logs/${dataset}_${variant}.log"
        
        echo "================================================================="
        echo " [$(date +'%Y-%m-%d %H:%M:%S')] 正在启动 -> Dataset: $dataset | Variant: $variant"
        echo " 日志将保存至: $LOG_FILE"
        echo "================================================================="
        
        # 执行训练命令 (注意这里去掉了 --pretrain_only 以便跑完联合微调全流程)
        python run_milan_sota.py \
            --dataset "$dataset" \
            --variant "$variant" 2>&1 | tee "$LOG_FILE"
            
        echo " [$(date +'%Y-%m-%d %H:%M:%S')] 完成 -> Dataset: $dataset | Variant: $variant"
        echo "-----------------------------------------------------------------"
        
    done
done

echo "🎉 所有数据集与消融实验已全部运行完毕！"