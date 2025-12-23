import os

# ================= 配置区域 =================

# 指定使用的显卡 ID (根据 gpustat 选择空闲的卡，例如 "2")
CUDA_DEVICE = "7"

# 基座模型路径 (Qwen3-4B-Instruct)
BASE_MODEL_PATH = "/data0/tygao/models/Qwen3-4B-Instruct-2507" # 请替换为你的Qwen3-4B实际路径或Huggingface ID

# 微调后的适配器路径 (DoRA Checkpoint)
ADAPTER_PATH = "/data0/dywang/Llm/Text2Sql/train_output/output_dora_optimized_all/v5-20251214-005056/checkpoint-3300" 

# 生成参数
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1  # 推理时温度低一点，保证SQL稳定性

# ================= 数据集配置 (核心修改) =================
# 格式说明:
# "path": 数据库文件夹绝对路径
# "mode": "folder" (代表 root/db_name/db_name.sqlite 结构)
#         "file"   (代表 root/db_name.sqlite 扁平结构)

DATASET_CONFIG = {
    "Spider (English)": {
        "path": "/data0/dywang/Llm/Text2Sql/spider/database",
        "mode": "folder"
    },
    "CSpider (Chinese)": {
        "path": "/data0/dywang/Llm/Text2Sql/cspider/database",
        "mode": "folder"
    },
    "Bird (English)": {
        "path": "/data0/dywang/Llm/Text2Sql/data_collected/Bird/dev/dev_databases",
        "mode": "folder"
    },
    "DuSQL (Chinese)": {
        "path": "/data0/tygao/classes/text2sql/evaluation/temp_databases/dusql_databases",
        "mode": "file" 
    }
}

# 默认数据集
DEFAULT_DATASET = "Spider (English)"