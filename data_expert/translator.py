"""
双语数据处理专家 - 双语翻译改写模块
Bilingual Translator for Text-to-SQL datasets
"""

import json
import logging
from typing import List, Dict, Any, Optional

from .api_client import QwenClient
from .prompts import (
    TRANSLATION_SYSTEM_PROMPT,
    TRANSLATION_USER_TEMPLATE,
    SCHEMA_LOCALIZE_SYSTEM_PROMPT,
    SCHEMA_LOCALIZE_USER_TEMPLATE,
    DIRTY_DATA_SYSTEM_PROMPT,
    DIRTY_DATA_USER_TEMPLATE
)
from .config import DEFAULT_MODEL

logger = logging.getLogger(__name__)


class BilingualTranslator:
    """双语翻译与改写器"""
    
    def __init__(self, client: QwenClient = None):
        """
        初始化双语翻译器
        
        Args:
            client: Qwen API客户端
        """
        self.client = client or QwenClient()
        self.model = DEFAULT_MODEL
    
    def translate_sample(
        self,
        question_en: str,
        sql: str,
        schema: Dict[str, Any],
        db_id: str,
        n_variants: int = 3
    ) -> List[Dict[str, Any]]:
        """
        翻译并改写单个样本
        
        Args:
            question_en: 英文问题
            sql: SQL查询
            schema: 数据库Schema
            db_id: 数据库ID
            n_variants: 生成变体数量
            
        Returns:
            翻译后的变体列表
        """
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        
        messages = [
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": TRANSLATION_USER_TEMPLATE.format(
                question_en=question_en,
                sql=sql,
                schema=schema_str,
                db_id=db_id,
                n=n_variants
            )}
        ]
        
        config = self.client.get_task_config("translation")
        result = self.client.call_with_json_output(messages, model=self.model, **config)
        
        # 确保结果是列表
        if isinstance(result, dict):
            result = [result]
        
        # 添加原始信息
        for item in result:
            item["original_question_en"] = question_en
            item["original_sql"] = sql
            item["db_id"] = db_id
        
        return result
    
    def translate_batch(
        self,
        samples: List[Dict[str, Any]],
        schema_dict: Dict[str, Dict[str, Any]],
        n_variants: int = 3
    ) -> List[Dict[str, Any]]:
        """
        批量翻译样本
        
        Args:
            samples: 样本列表
            schema_dict: db_id到schema的映射
            n_variants: 每个样本生成的变体数量
            
        Returns:
            翻译结果列表
        """
        all_results = []
        
        for i, sample in enumerate(samples):
            try:
                schema = schema_dict.get(sample["db_id"], {})
                question_en = sample.get("question", sample.get("question_en", ""))
                
                variants = self.translate_sample(
                    question_en=question_en,
                    sql=sample["sql"],
                    schema=schema,
                    db_id=sample["db_id"],
                    n_variants=n_variants
                )
                
                for v in variants:
                    v["source_index"] = i
                all_results.extend(variants)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"已翻译 {i + 1}/{len(samples)} 个样本，共生成 {len(all_results)} 条数据")
                    
            except Exception as e:
                logger.error(f"翻译样本 {i} 失败: {e}")
        
        return all_results
    
    def localize_schema(self, schema: Dict[str, Any], db_id: str) -> Dict[str, Any]:
        """
        Schema中文化
        
        Args:
            schema: 原始Schema
            db_id: 数据库ID
            
        Returns:
            中文化的Schema
        """
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        
        messages = [
            {"role": "system", "content": SCHEMA_LOCALIZE_SYSTEM_PROMPT},
            {"role": "user", "content": SCHEMA_LOCALIZE_USER_TEMPLATE.format(
                schema=schema_str,
                db_id=db_id
            )}
        ]
        
        config = self.client.get_task_config("translation")
        result = self.client.call_with_json_output(messages, model=self.model, **config)
        
        return result
    
    def generate_dirty_variants(
        self,
        question_zh: str,
        question_en: str,
        sql: str,
        db_id: str,
        n_variants: int = 3
    ) -> List[Dict[str, Any]]:
        """
        生成脏数据变体（用于增强模型鲁棒性）
        
        Args:
            question_zh: 中文问题
            question_en: 英文问题
            sql: SQL查询
            db_id: 数据库ID
            n_variants: 生成变体数量
            
        Returns:
            脏数据变体列表
        """
        messages = [
            {"role": "system", "content": DIRTY_DATA_SYSTEM_PROMPT},
            {"role": "user", "content": DIRTY_DATA_USER_TEMPLATE.format(
                question_zh=question_zh,
                question_en=question_en,
                sql=sql,
                db_id=db_id,
                n=n_variants
            )}
        ]
        
        config = self.client.get_task_config("synthesis")  # 用更高温度
        result = self.client.call_with_json_output(messages, model=self.model, **config)
        
        if isinstance(result, dict):
            result = [result]
        
        return result
    
    def paraphrase_question(
        self,
        question: str,
        language: str = "zh",
        n_variants: int = 3,
        style: str = "diverse"
    ) -> List[str]:
        """
        问题同义改写
        
        Args:
            question: 原始问题
            language: 语言 (zh/en)
            n_variants: 变体数量
            style: 改写风格 (diverse/formal/colloquial)
            
        Returns:
            改写后的问题列表
        """
        style_instructions = {
            "diverse": "生成多样化的表述，包括正式、口语、简洁等多种风格",
            "formal": "使用正式、书面的表达方式",
            "colloquial": "使用口语化、日常的表达方式"
        }
        
        lang_name = "中文" if language == "zh" else "English"
        
        prompt = f"""请将以下问题改写为{n_variants}个同义变体。

原始问题（{lang_name}）：{question}

要求：
- {style_instructions.get(style, style_instructions['diverse'])}
- 保持语义完全一致
- 每个变体的表述方式要明显不同

输出JSON列表格式：
["变体1", "变体2", ...]

只输出JSON列表。"""

        messages = [
            {"role": "system", "content": "你是一个语言改写专家，擅长生成语义相同但表述不同的句子变体。"},
            {"role": "user", "content": prompt}
        ]
        
        config = self.client.get_task_config("translation")
        result = self.client.call_with_json_output(messages, model=self.model, **config)
        
        return result if isinstance(result, list) else [result]
    
    def create_bilingual_dataset(
        self,
        samples: List[Dict[str, Any]],
        schema_dict: Dict[str, Dict[str, Any]],
        include_dirty: bool = True,
        n_clean_variants: int = 2,
        n_dirty_variants: int = 1
    ) -> List[Dict[str, Any]]:
        """
        创建完整的双语数据集
        
        Args:
            samples: 原始样本列表
            schema_dict: Schema字典
            include_dirty: 是否包含脏数据变体
            n_clean_variants: 干净变体数量
            n_dirty_variants: 脏数据变体数量
            
        Returns:
            双语数据集
        """
        dataset = []
        
        for i, sample in enumerate(samples):
            try:
                schema = schema_dict.get(sample["db_id"], {})
                question_en = sample.get("question", sample.get("question_en", ""))
                
                # 生成干净的翻译变体
                clean_variants = self.translate_sample(
                    question_en=question_en,
                    sql=sample["sql"],
                    schema=schema,
                    db_id=sample["db_id"],
                    n_variants=n_clean_variants
                )
                
                for v in clean_variants:
                    v["data_type"] = "clean"
                    v["source_index"] = i
                dataset.extend(clean_variants)
                
                # 生成脏数据变体
                if include_dirty and clean_variants:
                    first_variant = clean_variants[0]
                    dirty_variants = self.generate_dirty_variants(
                        question_zh=first_variant.get("question_zh", ""),
                        question_en=question_en,
                        sql=sample["sql"],
                        db_id=sample["db_id"],
                        n_variants=n_dirty_variants
                    )
                    
                    for v in dirty_variants:
                        v["data_type"] = "dirty"
                        v["source_index"] = i
                    dataset.extend(dirty_variants)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"已处理 {i + 1}/{len(samples)} 个样本")
                    
            except Exception as e:
                logger.error(f"处理样本 {i} 失败: {e}")
        
        return dataset


def demo_translation():
    """演示翻译功能"""
    translator = BilingualTranslator()
    
    # 示例样本
    sample = {
        "question": "Find the name of students who have grade higher than 80",
        "sql": "SELECT name FROM students WHERE grade > 80",
        "db_id": "school"
    }
    
    schema = {
        "tables": [
            {
                "name": "students",
                "columns": ["id", "name", "grade", "class_id"]
            }
        ]
    }
    
    results = translator.translate_sample(
        question_en=sample["question"],
        sql=sample["sql"],
        schema=schema,
        db_id=sample["db_id"],
        n_variants=3
    )
    
    print("翻译结果:")
    for r in results:
        print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo_translation()
