import os

# ================= 配置区域 =================

# 指定使用的显卡 ID (根据 gpustat 选择空闲的卡，例如 "2")
CUDA_DEVICE = os.getenv("BILINGUAL_SQL_CODER_CUDA_DEVICE", "0")

# 基座模型路径 (Qwen3.5-4B)
BASE_MODEL_PATH = os.getenv("BILINGUAL_SQL_CODER_MODEL_PATH", "Qwen/Qwen3.5-4B")

# 微调后的适配器路径 (DoRA/LoRA Checkpoint)
# 注意：Qwen3 adapter 通常不能直接加载到 Qwen3.5；这里默认使用基座模型。
# 如已重新基于 Qwen3.5 微调，可设置环境变量 BILINGUAL_SQL_CODER_ADAPTER_PATH。
ADAPTER_PATH = os.getenv("BILINGUAL_SQL_CODER_ADAPTER_PATH", "")

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
        "path": os.getenv("SPIDER_DB_PATH", os.path.join("data", "spider", "database")),
        "mode": "folder"
    },
    "CSpider (Chinese)": {
        "path": os.getenv("CSPIDER_DB_PATH", os.path.join("data", "cspider", "database")),
        "mode": "folder"
    },
    "Bird (English)": {
        "path": os.getenv("BIRD_DB_PATH", os.path.join("data_collected", "Bird", "dev", "dev_databases")),
        "mode": "folder"
    },
    "DuSQL (Chinese)": {
        "path": os.getenv("DUSQL_DB_PATH", os.path.join("evaluation", "temp_databases", "dusql_databases")),
        "mode": "file" 
    }
}

# 默认数据集
DEFAULT_DATASET = "Spider (English)"
