"""
双语数据处理专家 - 数据清洗模块
Data Cleaner for Text-to-SQL datasets
"""

import json
import logging
import sqlite3
from typing import List, Dict, Any, Optional
from pathlib import Path

from .api_client import QwenClient
from .prompts import (
    CLEANING_SYSTEM_PROMPT,
    CLEANING_USER_TEMPLATE,
    SQL_VALIDATION_SYSTEM_PROMPT,
    SQL_VALIDATION_USER_TEMPLATE
)
from .config import SQL_MODEL

logger = logging.getLogger(__name__)


class DataCleaner:
    """Text-to-SQL数据清洗器"""
    
    def __init__(self, client: QwenClient = None):
        """
        初始化数据清洗器
        
        Args:
            client: Qwen API客户端
        """
        self.client = client or QwenClient()
        self.model = SQL_MODEL
    
    def clean_sample(
        self,
        question: str,
        sql: str,
        schema: Dict[str, Any],
        db_id: str
    ) -> Dict[str, Any]:
        """
        清洗单个样本
        
        Args:
            question: 问题文本
            sql: SQL查询
            schema: 数据库Schema
            db_id: 数据库ID
            
        Returns:
            清洗结果
        """
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        
        messages = [
            {"role": "system", "content": CLEANING_SYSTEM_PROMPT},
            {"role": "user", "content": CLEANING_USER_TEMPLATE.format(
                question=question,
                sql=sql,
                schema=schema_str,
                db_id=db_id
            )}
        ]
        
        config = self.client.get_task_config("cleaning")
        result = self.client.call_with_json_output(messages, model=self.model, **config)
        
        return result
    
    def clean_batch(
        self,
        samples: List[Dict[str, Any]],
        schema_dict: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        批量清洗样本
        
        Args:
            samples: 样本列表，每个样本包含question, sql, db_id
            schema_dict: db_id到schema的映射
            
        Returns:
            清洗结果列表
        """
        results = []
        for i, sample in enumerate(samples):
            try:
                schema = schema_dict.get(sample["db_id"], {})
                result = self.clean_sample(
                    question=sample.get("question", sample.get("question_en", "")),
                    sql=sample["sql"],
                    schema=schema,
                    db_id=sample["db_id"]
                )
                result["original_index"] = i
                result["original_sample"] = sample
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"已清洗 {i + 1}/{len(samples)} 个样本")
                    
            except Exception as e:
                logger.error(f"清洗样本 {i} 失败: {e}")
                results.append({
                    "is_valid": None,
                    "error": str(e),
                    "original_index": i,
                    "original_sample": sample
                })
        
        return results
    
    def validate_sql_syntax(self, sql: str, db_path: str = None) -> Dict[str, Any]:
        """
        使用SQLite验证SQL语法
        
        Args:
            sql: SQL查询语句
            db_path: 数据库文件路径（可选）
            
        Returns:
            验证结果
        """
        result = {
            "is_valid": False,
            "error": None,
            "executed": False
        }
        
        try:
            # 使用内存数据库验证语法
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()
            
            # 只解析不执行
            cursor.execute(f"EXPLAIN {sql}")
            result["is_valid"] = True
            
            conn.close()
            
        except sqlite3.Error as e:
            result["error"] = str(e)
        
        # 如果提供了数据库路径，尝试实际执行
        if db_path and Path(db_path).exists():
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(sql)
                result["executed"] = True
                result["row_count"] = len(cursor.fetchall())
                conn.close()
            except sqlite3.Error as e:
                result["execution_error"] = str(e)
        
        return result
    
    def validate_with_llm(
        self,
        question: str,
        sql: str,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        使用LLM验证SQL正确性
        
        Args:
            question: 问题文本
            sql: SQL查询
            schema: 数据库Schema
            
        Returns:
            验证结果
        """
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        
        messages = [
            {"role": "system", "content": SQL_VALIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": SQL_VALIDATION_USER_TEMPLATE.format(
                question=question,
                sql=sql,
                schema=schema_str
            )}
        ]
        
        config = self.client.get_task_config("cleaning")
        result = self.client.call_with_json_output(messages, model=self.model, **config)
        
        return result
    
    def filter_valid_samples(
        self,
        samples: List[Dict[str, Any]],
        confidence_threshold: float = 0.8
    ) -> tuple:
        """
        过滤出有效样本
        
        Args:
            samples: 清洗后的样本列表
            confidence_threshold: 置信度阈值
            
        Returns:
            (有效样本列表, 无效样本列表)
        """
        valid = []
        invalid = []
        
        for sample in samples:
            if sample.get("is_valid") and sample.get("confidence", 0) >= confidence_threshold:
                valid.append(sample)
            else:
                invalid.append(sample)
        
        logger.info(f"过滤结果: 有效 {len(valid)}, 无效 {len(invalid)}")
        
        return valid, invalid
    
    def generate_cleaning_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        生成清洗报告
        
        Args:
            results: 清洗结果列表
            
        Returns:
            报告统计
        """
        total = len(results)
        valid_count = sum(1 for r in results if r.get("is_valid"))
        invalid_count = sum(1 for r in results if r.get("is_valid") is False)
        error_count = sum(1 for r in results if r.get("is_valid") is None)
        
        # 统计问题类型
        issue_types = {}
        for r in results:
            for issue in r.get("issues", []):
                issue_types[issue] = issue_types.get(issue, 0) + 1
        
        report = {
            "total_samples": total,
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "error_count": error_count,
            "valid_rate": valid_count / total if total > 0 else 0,
            "issue_distribution": issue_types,
            "avg_confidence": sum(r.get("confidence", 0) for r in results if r.get("confidence")) / max(valid_count, 1)
        }
        
        return report


def demo_cleaning():
    """演示数据清洗"""
    cleaner = DataCleaner()
    
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
    
    result = cleaner.clean_sample(
        question=sample["question"],
        sql=sample["sql"],
        schema=schema,
        db_id=sample["db_id"]
    )
    
    print("清洗结果:", json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo_cleaning()
