"""
双语数据处理专家 - 配置文件
Configuration for Bilingual Data Processing Expert
"""

import os

# ========================
# API 配置
# ========================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-xxxxxxxx")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ========================
# 模型配置
# ========================
MODELS = {
    "qwen3-max": {
        "id": "qwen3-max",
        "description": "最强推理、中英双语顶级、复杂SQL逻辑优秀",
        "use_case": "高质量种子生成、复杂清洗",
        "cost": "high",
        "speed": "slow"
    },
    "qwen-plus": {
        "id": "qwen-plus",
        "description": "性价比最高、中英平衡、指令遵循强",
        "use_case": "主力生成（推荐首选）",
        "cost": "medium",
        "speed": "medium"
    },
    "qwen3-coder-plus": {
        "id": "qwen3-coder-plus",
        "description": "代码/SQL专精、工具调用强",
        "use_case": "SQL相关生成、脏数据清洗",
        "cost": "medium",
        "speed": "medium"
    },
    "qwen-flash": {
        "id": "qwen-flash",
        "description": "速度快、成本低",
        "use_case": "批量扩展（10k+条）",
        "cost": "low",
        "speed": "fast"
    }
}

# 默认模型
DEFAULT_MODEL = "qwen-plus"
BATCH_MODEL = "qwen-flash"
COMPLEX_MODEL = "qwen3-max"
SQL_MODEL = "qwen3-coder-plus"

# ========================
# 生成参数配置
# ========================
GENERATION_CONFIG = {
    "translation": {
        "temperature": 0.7,
        "max_tokens": 2048,
        "top_p": 0.95
    },
    "synthesis": {
        "temperature": 0.9,
        "max_tokens": 4096,
        "top_p": 0.95
    },
    "cleaning": {
        "temperature": 0.3,
        "max_tokens": 4096,
        "top_p": 0.9
    },
    "evaluation": {
        "temperature": 1.0,
        "max_tokens": 2048,
        "top_p": 0.95
    }
}

# ========================
# 路径配置
# ========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "generated_data")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

# ========================
# 并发配置
# ========================
MAX_CONCURRENT_REQUESTS = 5
REQUEST_DELAY = 0.5  # 请求间隔（秒）
MAX_RETRIES = 3
RETRY_DELAY = 2  # 重试间隔（秒）

# ========================
# 数据配置
# ========================
SPIDER_DATASET_NAME = "spider"
DEFAULT_BATCH_SIZE = 10
