"""
双语数据处理专家 - 工具函数
Utility Functions for Bilingual Data Processing Expert
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], path: str):
    """保存JSONL文件"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_json(path: str) -> Any:
    """加载JSON文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: str):
    """保存JSON文件"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def hash_sample(sample: Dict[str, Any]) -> str:
    """生成样本的唯一哈希"""
    key_fields = ["question", "question_en", "question_zh", "sql", "db_id"]
    content = ""
    for field in key_fields:
        if field in sample:
            content += str(sample[field])
    return hashlib.md5(content.encode()).hexdigest()[:12]


def deduplicate_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """去重样本"""
    seen = set()
    unique = []
    for sample in samples:
        h = hash_sample(sample)
        if h not in seen:
            seen.add(h)
            unique.append(sample)
    return unique


def format_schema_for_prompt(schema: Dict[str, Any]) -> str:
    """格式化Schema用于Prompt"""
    lines = []
    tables = schema.get("tables", schema.get("table_names", []))
    
    if isinstance(tables, list) and tables:
        if isinstance(tables[0], dict):
            # 结构化格式
            for table in tables:
                table_name = table.get("name", table.get("table_name", ""))
                columns = table.get("columns", [])
                
                if isinstance(columns, list) and columns:
                    if isinstance(columns[0], dict):
                        col_strs = [f"{c.get('name', c.get('column_name', ''))} ({c.get('type', 'TEXT')})" 
                                   for c in columns]
                    else:
                        col_strs = columns
                    lines.append(f"表 {table_name}: {', '.join(col_strs)}")
                else:
                    lines.append(f"表 {table_name}")
        else:
            # 简单列表格式
            for table_name in tables:
                lines.append(f"表 {table_name}")
    
    return '\n'.join(lines) if lines else json.dumps(schema, ensure_ascii=False, indent=2)


def split_data(
    data: List[Dict[str, Any]],
    ratios: Dict[str, float],
    seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """
    按比例划分数据
    
    Args:
        data: 数据列表
        ratios: 划分比例，如 {"train": 0.8, "dev": 0.1, "test": 0.1}
        seed: 随机种子
        
    Returns:
        划分后的数据字典
    """
    import random
    random.seed(seed)
    
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    result = {}
    start = 0
    
    for name, ratio in ratios.items():
        n = int(len(shuffled) * ratio)
        result[name] = shuffled[start:start + n]
        start += n
    
    # 处理剩余数据
    if start < len(shuffled):
        last_key = list(ratios.keys())[-1]
        result[last_key].extend(shuffled[start:])
    
    return result


def merge_datasets(*datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """合并多个数据集并去重"""
    merged = []
    for dataset in datasets:
        merged.extend(dataset)
    return deduplicate_samples(merged)


def validate_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """验证样本格式"""
    issues = []
    
    # 检查必需字段
    required_fields = ["sql"]
    question_fields = ["question", "question_en", "question_zh"]
    
    for field in required_fields:
        if field not in sample or not sample[field]:
            issues.append(f"缺少必需字段: {field}")
    
    has_question = any(sample.get(f) for f in question_fields)
    if not has_question:
        issues.append("缺少问题字段 (question/question_en/question_zh)")
    
    # 检查SQL基本格式
    sql = sample.get("sql", "")
    if sql and not any(kw in sql.upper() for kw in ["SELECT", "INSERT", "UPDATE", "DELETE"]):
        issues.append("SQL格式可能不正确")
    
    return {
        "is_valid": len(issues) == 0,
        "issues": issues
    }


def batch_validate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """批量验证样本"""
    valid = []
    invalid = []
    
    for i, sample in enumerate(samples):
        result = validate_sample(sample)
        if result["is_valid"]:
            valid.append(sample)
        else:
            invalid.append({
                "index": i,
                "sample": sample,
                "issues": result["issues"]
            })
    
    return {
        "valid": valid,
        "invalid": invalid,
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "valid_rate": len(valid) / len(samples) if samples else 0
    }


def convert_spider_format(sample: Dict[str, Any]) -> Dict[str, Any]:
    """转换Spider格式到统一格式"""
    return {
        "question_en": sample.get("question", ""),
        "question_zh": sample.get("question_zh", ""),
        "sql": sample.get("query", sample.get("sql", "")),
        "db_id": sample.get("db_id", ""),
        "hardness": sample.get("hardness", "unknown")
    }


def convert_to_training_format(sample: Dict[str, Any], template: str = "default") -> Dict[str, Any]:
    """
    转换为训练格式
    
    Args:
        sample: 原始样本
        template: 模板类型
        
    Returns:
        训练格式的样本
    """
    templates = {
        "default": "根据以下数据库结构，将问题转换为SQL查询。\n\n问题：{question}\n\nSQL：",
        "simple": "问题：{question}\nSQL：",
        "detailed": "数据库：{db_id}\n结构：{schema}\n\n用户问题：{question}\n\n请生成对应的SQL查询："
    }
    
    prompt_template = templates.get(template, templates["default"])
    
    question = sample.get("question_zh") or sample.get("question_en") or sample.get("question", "")
    
    return {
        "instruction": prompt_template.format(
            question=question,
            db_id=sample.get("db_id", ""),
            schema=sample.get("schema", "")
        ),
        "input": "",
        "output": sample.get("sql", ""),
        "metadata": {
            "db_id": sample.get("db_id", ""),
            "difficulty": sample.get("difficulty", sample.get("hardness", "unknown"))
        }
    }


def export_for_lora_training(
    samples: List[Dict[str, Any]],
    output_path: str,
    template: str = "default"
):
    """导出为LoRA微调格式"""
    training_data = [convert_to_training_format(s, template) for s in samples]
    save_jsonl(training_data, output_path)
    return len(training_data)
