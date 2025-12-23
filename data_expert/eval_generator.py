"""
双语数据处理专家 - 评测数据生成模块
Evaluation Data Generator for Text-to-SQL datasets
"""

import json
import logging
import random
from typing import List, Dict, Any, Optional

from .api_client import QwenClient
from .prompts import (
    EVAL_SYSTEM_PROMPT,
    EVAL_USER_TEMPLATE
)
from .config import DEFAULT_MODEL

logger = logging.getLogger(__name__)


class EvalDataGenerator:
    """评测数据生成器"""
    
    def __init__(self, client: QwenClient = None):
        """
        初始化评测数据生成器
        
        Args:
            client: Qwen API客户端
        """
        self.client = client or QwenClient()
        self.model = DEFAULT_MODEL
    
    def generate_eval_variants(
        self,
        question: str,
        sql: str,
        schema: Dict[str, Any],
        db_id: str,
        n_variants: int = 3,
        original_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        生成评测数据变体
        
        Args:
            question: 原始问题
            sql: 原始SQL
            schema: 数据库Schema
            db_id: 数据库ID
            n_variants: 变体数量
            original_id: 原始样本ID
            
        Returns:
            评测变体列表
        """
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        original_id = original_id or f"{db_id}_{hash(question) % 10000}"
        
        messages = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": EVAL_USER_TEMPLATE.format(
                question=question,
                sql=sql,
                schema=schema_str,
                db_id=db_id,
                n=n_variants
            )}
        ]
        
        # 使用更高温度以增加多样性
        config = self.client.get_task_config("evaluation")
        result = self.client.call_with_json_output(messages, model=self.model, **config)
        
        if isinstance(result, dict):
            result = [result]
        
        for item in result:
            item["original_id"] = original_id
            item["db_id"] = db_id
            item["is_eval"] = True
        
        return result
    
    def generate_paraphrase_eval(
        self,
        question: str,
        sql: str,
        db_id: str,
        n_variants: int = 3
    ) -> List[Dict[str, Any]]:
        """
        生成同义改写评测数据（SQL不变，只改问题）
        
        Args:
            question: 原始问题
            sql: 原始SQL
            db_id: 数据库ID
            n_variants: 变体数量
            
        Returns:
            改写评测数据列表
        """
        prompt = f"""请将以下问题改写为{n_variants}个语义完全相同但表述不同的变体，用于评测模型的鲁棒性。

原始问题：{question}
原始SQL：{sql}

改写要求：
1. 语义必须100%一致（对应同一个SQL）
2. 表述方式要有明显差异：
   - 使用不同的句式（疑问句/陈述句/祈使句）
   - 使用同义词替换
   - 改变语序
   - 加入口语化表达
   - 使用不同的业务术语

输出JSON列表：
[
  {{
    "question_zh": "改写后的中文问题",
    "question_en": "Rewritten English question",
    "sql": "{sql}",
    "eval_type": "paraphrase"
  }},
  ...
]

只输出JSON列表。"""

        messages = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        config = self.client.get_task_config("evaluation")
        result = self.client.call_with_json_output(messages, model=self.model, **config)
        
        if isinstance(result, dict):
            result = [result]
        
        for item in result:
            item["db_id"] = db_id
            item["is_eval"] = True
            item["eval_type"] = "paraphrase"
        
        return result
    
    def generate_harder_variants(
        self,
        question: str,
        sql: str,
        schema: Dict[str, Any],
        db_id: str,
        n_variants: int = 2
    ) -> List[Dict[str, Any]]:
        """
        生成更难的变体（增加复杂度）
        
        Args:
            question: 原始问题
            sql: 原始SQL
            schema: 数据库Schema
            db_id: 数据库ID
            n_variants: 变体数量
            
        Returns:
            更难的变体列表
        """
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        
        prompt = f"""基于以下样本，生成{n_variants}个更难的变体，用于评测模型处理复杂查询的能力。

原始问题：{question}
原始SQL：{sql}
Schema：{schema_str}

增加难度的方式：
1. 增加更多筛选条件
2. 添加聚合函数或分组
3. 加入排序和限制
4. 使用更复杂的嵌套结构
5. 增加多表关联

要求：
- 新问题要自然、合理
- SQL必须语法正确
- 难度要明显高于原始样本

输出JSON列表：
[
  {{
    "question_zh": "更难的中文问题",
    "question_en": "Harder English question",
    "sql": "SELECT ...",
    "eval_type": "harder",
    "difficulty_increase": "描述增加了什么难度"
  }},
  ...
]

只输出JSON列表。"""

        messages = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        config = self.client.get_task_config("evaluation")
        result = self.client.call_with_json_output(messages, model=self.model, **config)
        
        if isinstance(result, dict):
            result = [result]
        
        for item in result:
            item["db_id"] = db_id
            item["is_eval"] = True
            item["eval_type"] = "harder"
        
        return result
    
    def generate_edge_cases(
        self,
        schema: Dict[str, Any],
        db_id: str,
        n_cases: int = 5
    ) -> List[Dict[str, Any]]:
        """
        生成边缘情况测试用例
        
        Args:
            schema: 数据库Schema
            db_id: 数据库ID
            n_cases: 用例数量
            
        Returns:
            边缘测试用例列表
        """
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        
        prompt = f"""基于以下Schema，生成{n_cases}个边缘情况测试用例，用于评测模型的鲁棒性。

Schema：{schema_str}
数据库ID：{db_id}

边缘情况类型：
1. 空结果查询（条件不可能满足）
2. 歧义表述（多种理解方式）
3. 复杂嵌套（多层子查询）
4. 特殊值处理（NULL、空字符串）
5. 边界条件（最大、最小、第一、最后）
6. 否定查询（NOT、除了、排除）
7. 复杂聚合（多重聚合、条件聚合）

输出JSON列表：
[
  {{
    "question_zh": "中文问题",
    "question_en": "English question",
    "sql": "SELECT ...",
    "eval_type": "edge_case",
    "edge_case_type": "边缘情况类型",
    "notes": "说明这个用例测试什么"
  }},
  ...
]

只输出JSON列表。"""

        messages = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        config = self.client.get_task_config("evaluation")
        result = self.client.call_with_json_output(messages, model=self.model, **config)
        
        if isinstance(result, dict):
            result = [result]
        
        for item in result:
            item["db_id"] = db_id
            item["is_eval"] = True
        
        return result
    
    def generate_eval_dataset(
        self,
        train_samples: List[Dict[str, Any]],
        schema_dict: Dict[str, Dict[str, Any]],
        holdout_ratio: float = 0.1,
        n_paraphrase: int = 2,
        n_harder: int = 1,
        include_edge_cases: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        生成完整的评测数据集
        
        Args:
            train_samples: 训练样本列表
            schema_dict: Schema字典
            holdout_ratio: 用于生成评测数据的样本比例
            n_paraphrase: 每个样本生成的改写变体数
            n_harder: 每个样本生成的更难变体数
            include_edge_cases: 是否包含边缘情况测试
            
        Returns:
            评测数据集字典
        """
        # 随机选取holdout样本
        n_holdout = max(1, int(len(train_samples) * holdout_ratio))
        holdout_indices = random.sample(range(len(train_samples)), n_holdout)
        holdout_samples = [train_samples[i] for i in holdout_indices]
        
        eval_dataset = {
            "paraphrase": [],
            "harder": [],
            "edge_cases": [],
            "holdout_indices": holdout_indices
        }
        
        # 为每个holdout样本生成变体
        for i, sample in enumerate(holdout_samples):
            db_id = sample["db_id"]
            schema = schema_dict.get(db_id, {})
            question = sample.get("question", sample.get("question_en", ""))
            sql = sample["sql"]
            
            try:
                # 生成改写变体
                paraphrases = self.generate_paraphrase_eval(
                    question, sql, db_id, n_paraphrase
                )
                eval_dataset["paraphrase"].extend(paraphrases)
                
                # 生成更难变体
                if schema:
                    harder = self.generate_harder_variants(
                        question, sql, schema, db_id, n_harder
                    )
                    eval_dataset["harder"].extend(harder)
                
                if (i + 1) % 5 == 0:
                    logger.info(f"已处理 {i + 1}/{len(holdout_samples)} 个样本")
                    
            except Exception as e:
                logger.error(f"生成评测数据失败 (样本 {i}): {e}")
        
        # 生成边缘情况测试
        if include_edge_cases:
            for db_id, schema in schema_dict.items():
                try:
                    edge_cases = self.generate_edge_cases(schema, db_id, n_cases=3)
                    eval_dataset["edge_cases"].extend(edge_cases)
                except Exception as e:
                    logger.error(f"生成边缘情况失败 (db: {db_id}): {e}")
        
        # 统计
        total = sum(len(v) for k, v in eval_dataset.items() if isinstance(v, list))
        logger.info(f"评测数据集生成完成: 共 {total} 条")
        logger.info(f"  - 改写变体: {len(eval_dataset['paraphrase'])}")
        logger.info(f"  - 更难变体: {len(eval_dataset['harder'])}")
        logger.info(f"  - 边缘情况: {len(eval_dataset['edge_cases'])}")
        
        return eval_dataset
    
    def split_train_eval(
        self,
        samples: List[Dict[str, Any]],
        eval_ratio: float = 0.1,
        seed: int = 42
    ) -> tuple:
        """
        划分训练集和评测集（确保无数据泄露）
        
        Args:
            samples: 全部样本
            eval_ratio: 评测集比例
            seed: 随机种子
            
        Returns:
            (训练集, 评测集)
        """
        random.seed(seed)
        
        # 按db_id分组
        db_samples = {}
        for sample in samples:
            db_id = sample.get("db_id", "unknown")
            if db_id not in db_samples:
                db_samples[db_id] = []
            db_samples[db_id].append(sample)
        
        train_set = []
        eval_set = []
        
        # 每个数据库按比例划分
        for db_id, db_data in db_samples.items():
            random.shuffle(db_data)
            n_eval = max(1, int(len(db_data) * eval_ratio))
            eval_set.extend(db_data[:n_eval])
            train_set.extend(db_data[n_eval:])
        
        logger.info(f"数据划分完成: 训练集 {len(train_set)}, 评测集 {len(eval_set)}")
        
        return train_set, eval_set


def demo_eval_generation():
    """演示评测数据生成"""
    generator = EvalDataGenerator()
    
    # 示例样本
    sample = {
        "question": "查找成绩高于80分的学生姓名",
        "sql": "SELECT name FROM students WHERE score > 80",
        "db_id": "school"
    }
    
    schema = {
        "tables": [
            {
                "name": "students",
                "columns": ["id", "name", "score", "class_id"]
            }
        ]
    }
    
    # 生成改写变体
    paraphrases = generator.generate_paraphrase_eval(
        sample["question"],
        sample["sql"],
        sample["db_id"],
        n_variants=3
    )
    
    print("改写变体:")
    for p in paraphrases:
        print(json.dumps(p, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo_eval_generation()
