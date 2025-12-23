"""
双语数据处理专家 - Bilingual Data Processing Expert
用于Text-to-SQL任务的数据清洗、翻译、扩展和生成

主要功能：
1. 数据清洗 (DataCleaner)
2. 双语翻译改写 (BilingualTranslator)
3. 合成数据生成 (DataSynthesizer)
4. 评测数据生成 (EvalDataGenerator)
"""

from .api_client import QwenClient, QuotaExhaustedError, APIServiceError
from .data_cleaner import DataCleaner
from .translator import BilingualTranslator
from .synthesizer import DataSynthesizer
from .eval_generator import EvalDataGenerator
from .pipeline import DataExpertPipeline

__version__ = "1.0.0"
__author__ = "Text2SQL Data Expert"

__all__ = [
    "QwenClient",
    "QuotaExhaustedError",
    "APIServiceError",
    "DataCleaner",
    "BilingualTranslator",
    "DataSynthesizer",
    "EvalDataGenerator",
    "DataExpertPipeline"
]
