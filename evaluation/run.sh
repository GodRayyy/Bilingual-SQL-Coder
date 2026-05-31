#!/bin/bash
# =============================================================================
# Text2SQL 多数据集评测脚本
# =============================================================================
# 
# 功能说明：
#   本脚本用于评测Text2SQL大模型在7个数据集上的性能
#   支持的数据集：Spider, Bird, WikiSQL (英文) | CSpider, Chase, DuSQL, AntSQL (中文)
#   
# 评测流程：
#   1. 加载微调后的模型（或基础模型）
#   2. 对每个数据集生成SQL预测
#   3. 评测生成的SQL（Exact Match + Execution Accuracy）
#   4. 输出每个数据集的评分和整体平均分
#
# 评测支持情况：
#   ✅ 完整评测（推理+Exact Match+Execution）：Spider, CSpider, Bird, DuSQL, Chase
#   ⚠️  仅推理：WikiSQL, AntSQL（这两个数据集的SQL是结构化格式，暂不支持直接评测）
#
# 使用方法：
#   bash run.sh                          # 运行默认配置
#   或修改下方参数后运行
#
# =============================================================================

# 指定使用哪几张显卡 (例如 "0" 或 "0,1,2,3")
export CUDA_VISIBLE_DEVICES=4
BASE_MODEL_ID="${BILINGUAL_SQL_CODER_MODEL_PATH:-Qwen/Qwen3.5-4B}"
QWEN35_ADAPTER_DIR="${BILINGUAL_SQL_CODER_ADAPTER_PATH:-}"

# ==================== 评测所有7个数据集 ====================
# 评测 Qwen3.5 微调模型 - 所有数据集（推理+评测）
# 说明：
#   - Spider, CSpider, Bird, DuSQL, Chase: 完整评测（Exact Match + Execution Accuracy）
#   - WikiSQL, AntSQL: 仅推理（SQL格式为结构化数据，暂不支持直接评测）
# python run_full_evaluation.py \
#     --model_type tuned \
#     --base_model_id "$BASE_MODEL_ID" \
#     --checkpoint_dir "$QWEN35_ADAPTER_DIR" \
#     --datasets all \
#     --etype all

# ==================== 其他示例 ====================

# 1. 默认评测 Qwen3.5 基座模型（Spider, CSpider, Bird, DuSQL）
python run_full_evaluation.py \
    --model_type base \
    --base_model_id "$BASE_MODEL_ID" \
    --datasets Spider,CSpider,Bird,DuSQL \
    --etype all

# # 2. 评测基于 Qwen3.5 重新微调的模型（Spider,CSpider,Bird,DuSQL）
# python run_full_evaluation.py \
#     --model_type tuned \
#     --base_model_id "$BASE_MODEL_ID" \
#     --checkpoint_dir "$QWEN35_ADAPTER_DIR" \
#     --datasets Spider,CSpider,Bird,DuSQL \
#     --etype all

# # 3. 只做推理，不评测
# python run_full_evaluation.py \
#     --model_type tuned \
#     --base_model_id "$BASE_MODEL_ID" \
#     --checkpoint_dir "$QWEN35_ADAPTER_DIR" \
#     --datasets Bird,WikiSQL,Chase,DuSQL,AntSQL \
#     --etype all

# # 4. 跳过推理，直接评测已有结果
# python run_full_evaluation.py \
#     --skip_inference \
#     --datasets Spider,CSpider,Bird,DuSQL \
#     --etype all

# # 5. 只评测英文数据集
# python run_full_evaluation.py \
#     --model_type tuned \
#     --base_model_id "$BASE_MODEL_ID" \
#     --checkpoint_dir "$QWEN35_ADAPTER_DIR" \
#     --datasets Spider,Bird \
#     --etype all

# # 6. 只评测中文数据集
# python run_full_evaluation.py \
#     --model_type tuned \
#     --base_model_id "$BASE_MODEL_ID" \
#     --checkpoint_dir "$QWEN35_ADAPTER_DIR" \
#     --datasets CSpider,Chase,DuSQL \
#     --etype all
