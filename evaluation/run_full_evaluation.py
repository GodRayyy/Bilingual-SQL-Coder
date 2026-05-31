#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text2SQL 多数据集评测系统 - 主评测脚本
===========================================

功能：
    对Text2SQL微调模型进行全面评测，支持7个主流数据集

支持的数据集：
    英文: Spider, Bird, WikiSQL
    中文: CSpider, Chase, DuSQL, AntSQL

评测流程：
    1. 加载微调模型（或基础模型）
    2. 对每个数据集生成SQL预测
    3. 评测生成的SQL（Exact Match + Execution Accuracy）
    4. 输出每个数据集和整体的评分

使用示例：
    # 评测所有数据集
    python run_full_evaluation.py --model_type tuned --checkpoint_dir /path/to/checkpoint --datasets all
    
    # 只评测Spider和CSpider
    python run_full_evaluation.py --model_type tuned --checkpoint_dir /path/to/checkpoint --datasets Spider,CSpider
    
    # 使用基础模型评测
    python run_full_evaluation.py --model_type base --datasets all

注意事项：
    - WikiSQL和AntSQL的SQL是结构化格式，暂时只支持推理，不支持自动评测
    - Chase需要自动生成gold SQL文件（从多轮对话中提取第一轮）
    - 评测需要GPU支持
"""

import os
import sys
import json
import argparse
import torch
import re
import pandas as pd
from io import StringIO
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from transformers import AutoProcessor, BitsAndBytesConfig
try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

# 尝试导入 swift，如果失败则使用 transformers 的 set_seed
try:
    from swift.utils import seed_everything
except ImportError:
    from transformers import set_seed as seed_everything

# ================= 配置路径 =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.getenv("BILINGUAL_SQL_CODER_DATA_ROOT", PROJECT_ROOT)
DATA_COLLECTED_DIR = os.path.join(BASE_DIR, "data_collected")
EVAL_SCRIPT_DIR = os.path.join(DATA_COLLECTED_DIR, "spider/eval")
BASE_MODEL_ID = os.getenv("BILINGUAL_SQL_CODER_MODEL_PATH", "Qwen/Qwen3.5-4B")

# 数据集配置字典 - 支持7个数据集
DATASET_CONFIGS = {
    # ========== 英文数据集 ==========
    "Spider": {
        "language": "en",
        "dev_file": os.path.join(DATA_COLLECTED_DIR, "spider/dev.json"),
        "tables_file": os.path.join(DATA_COLLECTED_DIR, "spider/tables.json"),
        "db_dir": os.path.join(DATA_COLLECTED_DIR, "spider/database"),
        "gold_sql": os.path.join(DATA_COLLECTED_DIR, "spider/gt_sql/dev_gold.sql"),
        "has_evaluator": True  # 使用通用评测器
    },
    "Bird": {
        "language": "en",
        "dev_file": os.path.join(DATA_COLLECTED_DIR, "Bird/dev/dev.json"),
        "tables_file": os.path.join(DATA_COLLECTED_DIR, "Bird/dev/dev_tables.json"),
        "db_dir": os.path.join(DATA_COLLECTED_DIR, "Bird/dev/dev_databases"),
        "gold_sql": os.path.join(DATA_COLLECTED_DIR, "Bird/dev/dev.sql"),
        "has_evaluator": True  # 使用通用评测器
    },
    "WikiSQL": {
        "language": "en",
        "dev_file": os.path.join(DATA_COLLECTED_DIR, "WikiSQL/data/dev.jsonl"),
        "tables_file": os.path.join(DATA_COLLECTED_DIR, "WikiSQL/data/dev.tables.jsonl"),
        "db_dir": None,  # WikiSQL使用单个SQLite文件，不适合通用评测器的execution测试
        "gold_sql": None,  # 将动态生成到evaluation目录
        "has_evaluator": True,  # 使用通用评测器（需要先转换结构化SQL）
        "is_jsonl": True,
        "needs_gold_generation": True,  # 需要从结构化SQL生成gold SQL
        "gold_generation_type": "wikisql"  # 指定生成类型
    },
    
    # ========== 中文数据集 ==========
    "CSpider": {
        "language": "zh",
        "dev_file": os.path.join(DATA_COLLECTED_DIR, "CSpider/dev.json"),
        "tables_file": os.path.join(DATA_COLLECTED_DIR, "CSpider/tables.json"),
        "db_dir": os.path.join(DATA_COLLECTED_DIR, "CSpider/database"),
        "gold_sql": os.path.join(DATA_COLLECTED_DIR, "CSpider/dev_gold.sql"),
        "has_evaluator": True  # 使用通用评测器
    },
    "Chase": {
        "language": "zh",
        "dev_file": os.path.join(DATA_COLLECTED_DIR, "chase/data/dev.json"),
        "tables_file": os.path.join(DATA_COLLECTED_DIR, "chase/data/tables.json"),
        "db_dir": None,  # 将动态创建
        "gold_sql": None,  # 将动态生成到evaluation目录
        "has_evaluator": True,  # 使用通用评测器
        "is_multi_turn": True,  # Chase 是多轮对话格式
        "needs_gold_generation": True,  # 需要生成gold SQL
        "needs_db_building": True,  # 需要从JSON构建数据库
        "db_type": "chase"  # 数据库类型
    },
    "DuSQL": {
        "language": "zh",
        "dev_file": os.path.join(DATA_COLLECTED_DIR, "DuSQL/dev.json"),
        "tables_file": os.path.join(DATA_COLLECTED_DIR, "DuSQL/db_schema.json"),
        "db_dir": os.getenv(
            "DUSQL_DB_PATH",
            os.path.join(PROJECT_ROOT, "evaluation", "temp_databases", "dusql_databases")
        ),  # 使用已构建的数据库
        "db_schema_file": os.path.join(DATA_COLLECTED_DIR, "DuSQL/db_schema.json"),
        "db_content_file": os.path.join(DATA_COLLECTED_DIR, "DuSQL/db_content.json"),
        "gold_sql": os.path.join(DATA_COLLECTED_DIR, "DuSQL/gold_dev.sql"),
        "has_evaluator": True,  # 使用通用评测器
        "needs_db_building": False,  # 数据库已存在，不需要重新构建
        "db_type": "dusql"  # 数据库类型
    },
    "AntSQL": {
        "language": "zh",
        "dev_file": os.path.join(DATA_COLLECTED_DIR, "antsql1/antsql1_dev.jsonl"),
        "tables_file": os.path.join(DATA_COLLECTED_DIR, "antsql1/antsql1_fundTable.xlsx"),
        "db_dir": None,  # AntSQL没有数据库文件，只支持Exact Match评测
        "gold_sql": None,  # 将动态生成到evaluation目录
        "has_evaluator": True,  # 使用通用评测器（需要先转换结构化SQL）
        "is_jsonl": True,
        "needs_gold_generation": True,  # 需要从结构化SQL生成gold SQL
        "gold_generation_type": "antsql"  # 指定生成类型
    }
}

# 导入通用评测脚本
try:
    from universal_evaluation import UniversalEvaluator
    print("✅ 成功导入通用评测模块")
except ImportError:
    print("⚠️  警告: 无法导入通用评测模块，部分评测功能可能不可用")
    UniversalEvaluator = None

# 导入JSON数据库构建工具
try:
    from json_db_builder import JSONDatabaseBuilder
    print("✅ 成功导入JSON数据库构建模块")
except ImportError:
    print("⚠️  警告: 无法导入JSON数据库构建模块，DuSQL和Chase的execution评测可能不可用")
    JSONDatabaseBuilder = None

# 导入结构化SQL转换工具
try:
    from structured_sql_converter import StructuredSQLConverter
    print("✅ 成功导入结构化SQL转换模块")
except ImportError:
    print("⚠️  警告: 无法导入结构化SQL转换模块，WikiSQL和AntSQL的评测可能不可用")
    StructuredSQLConverter = None

# ================= 辅助函数 =================

def load_tables(tables_path):
    """
    加载数据库 Schema 信息，并将其整理为易于查询的字典格式。
    支持多种格式：Spider格式JSON、WikiSQL JSONL、DuSQL格式、Excel
    """
    print(f"Loading tables from {tables_path}...")
    
    schema_map = {}
    
    # 处理 Excel 文件 (AntSQL)
    if tables_path.endswith('.xlsx') or tables_path.endswith('.xls'):
        try:
            df = pd.read_excel(tables_path)
            cols = df.columns.tolist()
            col_str = ", ".join([str(c) for c in cols])
            schema_map['antsql_default'] = f"Table fund_table, columns = [{col_str}]"
            return schema_map
        except Exception as e:
            print(f"Warning: Failed to load Excel file: {e}")
            return schema_map
    
    # 处理 JSONL 文件 (WikiSQL)
    if tables_path.endswith('.jsonl'):
        try:
            with open(tables_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.load(StringIO(line))
                        table_id = item.get('id', item.get('table_id', ''))
                        if 'header' in item:
                            cols_str = ", ".join(item['header'])
                            schema_map[table_id] = f"Table {table_id}, columns = [{cols_str}]"
            return schema_map
        except Exception as e:
            print(f"Warning: Failed to load JSONL file: {e}")
            return schema_map
    
    # 处理标准 JSON 文件
    with open(tables_path, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)
    
    # 处理 DuSQL 特殊格式
    if isinstance(tables_data, dict) and 'db_id' not in (tables_data if isinstance(tables_data, dict) else tables_data[0] if tables_data else {}):
        # DuSQL 格式: {db_id: {table_info}}
        for db_id, db_info in tables_data.items():
            if 'table_names' in db_info and 'column_names' in db_info:
                table_names = db_info['table_names']
                column_names = db_info['column_names']
                
                tables_dict = {}
                for idx, t_name in enumerate(table_names):
                    tables_dict[idx] = {'name': t_name, 'cols': []}
                
                for col_info in column_names:
                    if isinstance(col_info, list) and len(col_info) >= 2 and col_info[0] >= 0:
                        tables_dict[col_info[0]]['cols'].append(col_info[1])
                
                lines = []
                for t_idx, info in tables_dict.items():
                    col_str = ", ".join(info['cols'])
                    lines.append(f"Table {info['name']}, columns = [{col_str}]")
                schema_map[db_id] = "\n".join(lines)
        return schema_map
    
    # 处理标准 Spider 格式
    for db in tables_data:
        db_id = db.get('db_id', db.get('database_id'))
        if not db_id:
            continue
        
        table_names = db.get('table_names_original', db.get('table_names'))
        column_names = db.get('column_names_original', db.get('column_names'))
        
        if not table_names or not column_names:
            continue

        tables_dict = {} 
        for idx, t_name in enumerate(table_names):
            tables_dict[idx] = {'name': t_name, 'cols': []}
            
        for col_idx, col_info in enumerate(column_names):
            if isinstance(col_info, list) and len(col_info) >= 2:
                table_idx, col_name = col_info[0], col_info[1]
                if table_idx >= 0:
                    tables_dict[table_idx]['cols'].append(col_name)
        
        schema_lines = []
        for table_idx, info in tables_dict.items():
            t_name = info['name']
            c_str = ", ".join(info['cols'])
            schema_lines.append(f"Table {t_name}, columns = [{c_str}]")
            
        schema_str = "\n".join(schema_lines)
        schema_map[db_id] = schema_str
        
    return schema_map

def find_latest_checkpoint(checkpoint_dir):
    """支持直接指定checkpoint目录 + 适配.safetensors格式LoRA权重"""
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint目录不存在: {checkpoint_dir}")
    
    required_files = ['adapter_config.json', 'adapter_model.safetensors']
    has_lora_files = all(os.path.exists(os.path.join(checkpoint_dir, f)) for f in required_files)
    
    if has_lora_files:
        print(f"✅ 验证通过：传入目录是有效LoRA checkpoint")
        return checkpoint_dir
    
    subdirs = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    checkpoints = [d for d in subdirs if 'checkpoint' in d.lower()]
    
    if not checkpoints:
        raise FileNotFoundError(f"❌ 错误：传入目录无效或未找到checkpoint子目录: {checkpoint_dir}")
    
    latest_ckpt = max(checkpoints, key=os.path.getmtime)
    print(f"✅ 从父目录找到最新checkpoint：{latest_ckpt}")
    return latest_ckpt


def get_text_tokenizer(processor):
    """Return the text tokenizer from a Qwen3.5 processor."""
    return getattr(processor, "tokenizer", processor)


def prepare_qwen35_inputs(processor, messages, device, max_length=4096):
    """Build text-only Qwen3.5 inputs with the native chat template."""
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            enable_thinking=False,
        )
    except TypeError:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
    return inputs.to(device)


def decode_qwen35_response(processor, generated_ids):
    """Decode generated token ids with either processor or tokenizer APIs."""
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def clean_sql_output(response):
    """Remove Qwen thinking tags and Markdown fences from generated SQL."""
    cleaned_sql = response.strip()
    if "</think>" in cleaned_sql:
        cleaned_sql = cleaned_sql.split("</think>", 1)[1].strip()
    cleaned_sql = cleaned_sql.replace("<think>", "").replace("</think>", "").strip()
    if "```sql" in cleaned_sql:
        cleaned_sql = cleaned_sql.split("```sql", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned_sql:
        cleaned_sql = cleaned_sql.split("```", 1)[1].split("```", 1)[0].strip()
    return cleaned_sql.replace('\n', ' ').strip()


def load_qwen35_model(base_model_id, model_type, checkpoint_dir):
    """Load Qwen3.5 base model and optional Qwen3.5 LoRA/DoRA adapter."""
    if AutoModelForImageTextToText is None:
        raise ImportError(
            "当前 transformers 版本不支持 Qwen3.5。"
            "请升级到支持 AutoModelForImageTextToText 的版本（建议 >=4.57）。"
        )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    model_kwargs = {
        'device_map': 'auto',
        'dtype': torch.bfloat16,
        'quantization_config': bnb_config,
        'trust_remote_code': True,
        'low_cpu_mem_usage': True
    }

    print("\n=== 加载 Qwen3.5 模型 ===")
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(base_model_id, **model_kwargs)

    if model_type == 'tuned':
        if not checkpoint_dir:
            raise ValueError("❌ 使用tuned模式必须指定Qwen3.5 LoRA/DoRA权重目录")

        ckpt_path = find_latest_checkpoint(checkpoint_dir)
        print(f"LoRA/DoRA权重路径：{ckpt_path}")

        _ = PeftConfig.from_pretrained(ckpt_path)
        model = PeftModel.from_pretrained(
            model,
            ckpt_path,
            device_map='auto',
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print("✅ 微调模型（Qwen3.5基座+LoRA/DoRA）加载完成")
    else:
        print("✅ Qwen3.5基础模型加载完成")

    model.eval()
    return model, processor

# ================= 推理逻辑 =================

def generate_gold_sql_for_chase(dev_file, output_gold_file):
    """从Chase的dev.json生成gold SQL文件（只取第一轮对话）"""
    print(f"正在为Chase生成gold SQL文件...")
    with open(dev_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gold_sqls = []
    for item in data:
        if 'interaction' in item and isinstance(item['interaction'], list) and len(item['interaction']) > 0:
            first_turn = item['interaction'][0]
            sql = first_turn.get('query', first_turn.get('sql', ''))
            gold_sqls.append(sql)
        else:
            gold_sqls.append('')
    
    with open(output_gold_file, 'w', encoding='utf-8') as f:
        for sql in gold_sqls:
            f.write(sql + '\n')
    print(f"✅ Chase gold SQL已生成: {output_gold_file}")
    return output_gold_file

def generate_gold_sql_from_structured(dev_file, tables_file, output_gold_file, generation_type):
    """从结构化SQL生成gold SQL文件（WikiSQL和AntSQL）"""
    if StructuredSQLConverter is None:
        print("❌ 错误：结构化SQL转换模块未加载")
        return None
    
    converter = StructuredSQLConverter()
    
    try:
        if generation_type == 'wikisql':
            converter.convert_wikisql_file(dev_file, tables_file, output_gold_file)
        elif generation_type == 'antsql':
            converter.convert_antsql_file(dev_file, tables_file, output_gold_file)
        else:
            print(f"❌ 错误：未知的生成类型: {generation_type}")
            return None
        
        return output_gold_file
    except Exception as e:
        print(f"❌ 生成gold SQL失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_dev_data(dev_file, is_jsonl=False, is_multi_turn=False):
    """加载开发集数据，支持JSON和JSONL格式"""
    with open(dev_file, 'r', encoding='utf-8') as f:
        if is_jsonl:
            data = []
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        else:
            data = json.load(f)
    
    # 如果是多轮对话格式，只取第一轮（或展开所有轮次）
    if is_multi_turn:
        expanded_data = []
        for item in data:
            if 'interaction' in item and isinstance(item['interaction'], list):
                # 只取第一轮
                first_turn = item['interaction'][0]
                expanded_item = {
                    'db_id': item.get('db_id', ''),
                    'question': first_turn.get('utterance', first_turn.get('question', '')),
                    'query': first_turn.get('query', first_turn.get('sql', ''))
                }
                expanded_data.append(expanded_item)
            else:
                expanded_data.append(item)
        return expanded_data
    
    return data

def run_inference(model_type, checkpoint_dir, output_file, dataset_name, dataset_config, base_model_id=BASE_MODEL_ID):
    """统一的推理函数，支持所有数据集"""
    seed_everything(42)
    
    language = dataset_config.get('language', 'en')
    dev_file = dataset_config['dev_file']
    tables_file = dataset_config['tables_file']
    is_jsonl = dataset_config.get('is_jsonl', False)
    is_multi_turn = dataset_config.get('is_multi_turn', False)
    
    print(f"\n=== 开始推理 [{dataset_name}] ({language.upper()}) ===")
    print(f"基座模型路径：{base_model_id}")
    print(f"模型类型：{model_type}")
    if model_type == 'tuned':
        print(f"LoRA checkpoint目录：{checkpoint_dir}")

    model, processor = load_qwen35_model(base_model_id, model_type, checkpoint_dir)
    tokenizer = get_text_tokenizer(processor)

    # 准备数据
    print(f"加载数据库表结构：{tables_file}")
    schema_map = load_tables(tables_file)
    print(f"加载开发集数据：{dev_file}")
    dev_data = load_dev_data(dev_file, is_jsonl=is_jsonl, is_multi_turn=is_multi_turn)
    
    predictions = []
    print(f"\n开始生成SQL... (共 {len(dev_data)} 条)")
    
    for item in tqdm(dev_data, desc="生成SQL"):
        db_id = item.get('db_id', item.get('database_id', item.get('table_id', '')))
        question = item.get('question', item.get('question_text', ''))
        
        # 特殊处理：WikiSQL使用table_id，AntSQL使用默认db
        if not db_id or db_id == '':
            if 'table_id' in item:
                db_id = item['table_id']
            elif dataset_name == 'AntSQL':
                db_id = 'antsql_default'
            elif dataset_name == 'Chase' and not db_id:
                # Chase可能缺少db_id，从第一个可用的schema取
                if schema_map:
                    db_id = list(schema_map.keys())[0]
        
        if not question:
            predictions.append("SELECT * FROM T")
            continue
        
        if db_id not in schema_map:
            # 如果找不到对应的schema，使用第一个可用的
            if schema_map:
                db_id = list(schema_map.keys())[0]
            else:
                predictions.append("SELECT * FROM T")
                continue
            
        schema_context = schema_map[db_id]
        
        # 根据语言选择不同的 Prompt
        if language == "zh":
            # 中文 Prompt
            system_content = "你是一名专业的SQL数据分析师。" \
                             "请根据给定的数据库表结构（Schema）和用户提出的自然语言问题，" \
                             "生成一句有效的SQL查询语句。不要提供任何解释，只输出SQL代码。"
            user_content = f"数据库表结构 (Database Schema):\n{schema_context}\n\n问题 (Question): {question}\n\nSQL:"
        else:
            # 英文 Prompt
            system_content = "You are a professional SQL data analyst. " \
                             "Given a database schema and a natural language question, " \
                             "generate a valid SQL query. Do not provide any explanation, only the SQL."
            user_content = f"Database Schema:\n{schema_context}\n\nQuestion: {question}\n\nSQL:"
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        inputs = prepare_qwen35_inputs(processor, messages, model.device, max_length=4096)
        input_ids = inputs['input_ids']

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.01,
                do_sample=False,
            )
        
        input_len = input_ids.shape[1]
        output_ids = generated_ids[0][input_len:]
        response = decode_qwen35_response(processor, [output_ids])
        cleaned_sql = clean_sql_output(response)
        
        predictions.append(cleaned_sql)
    
    print(f"\n推理完成！共生成 {len(predictions)} 条SQL语句")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sql in predictions:
            f.write(sql + '\n')
    print(f"预测结果已保存到：{output_file}")
    
    return len(predictions)

# ================= 评测逻辑 =================

def parse_evaluation_output(output_text):
    """解析评测脚本的输出，提取关键指标"""
    results = {}
    
    # 提取 Exact Match
    em_match = re.search(r'exact[_ ]?match.*?:\s*([\d.]+)', output_text, re.IGNORECASE)
    if em_match:
        results['exact_match'] = float(em_match.group(1))
    
    # 提取 Execution Accuracy
    exec_match = re.search(r'(?:execution|exec).*?(?:accuracy|score).*?:\s*([\d.]+)', output_text, re.IGNORECASE)
    if exec_match:
        results['execution_accuracy'] = float(exec_match.group(1))
    
    # 提取难度级别分数
    for level in ['easy', 'medium', 'hard', 'extra']:
        level_match = re.search(rf'{level}.*?:\s*([\d.]+)', output_text, re.IGNORECASE)
        if level_match:
            results[level] = float(level_match.group(1))
    
    return results

def run_evaluation(gold_file, pred_file, db_dir, tables_file, etype, dataset_name="Spider"):
    """运行单个数据集的评测并返回结果（使用通用评测器）"""
    print(f"\n{'='*60}")
    print(f"开始运行 {dataset_name} 评测...")
    print(f"{'='*60}")
    
    if UniversalEvaluator is None:
        print("❌ 错误：通用评测模块未加载")
        return {}
    
    print(f"Gold SQL: {gold_file if gold_file else 'N/A (只计算Execution Accuracy)'}")
    print(f"Pred SQL: {pred_file}")
    print(f"Database: {db_dir}")
    print(f"Tables: {tables_file}")
    print(f"Eval Type: {etype}\n")
    
    # 创建评测器并执行评测
    evaluator = UniversalEvaluator(dataset_name=dataset_name)
    results = evaluator.evaluate(
        pred_file=pred_file,
        gold_file=gold_file,  # 可以为None
        db_dir=db_dir,
        tables_file=tables_file  # 传入tables文件用于加载外键映射
    )
    
    # 打印结果
    evaluator.print_results(results)
    
    return results

# ================= 主程序 =================

def main():
    parser = argparse.ArgumentParser(description="Text2Sql 整合评测脚本 (支持 7 个数据集)")
    
    # 推理参数
    parser.add_argument("--model_type", type=str, choices=['base', 'tuned'], default='tuned', help="模型类型: base 或 tuned")
    parser.add_argument("--base_model_id", type=str, default=BASE_MODEL_ID, help="Qwen3.5基座模型路径或Hugging Face ID")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="LoRA微调权重目录 (tuned模式必填)")
    parser.add_argument("--skip_inference", action="store_true", help="跳过推理，直接使用已有的输出文件进行评测")
    parser.add_argument("--datasets", type=str, default="Spider,CSpider", help="要测试的数据集，逗号分隔 (支持: Spider,CSpider,Bird,WikiSQL,Chase,DuSQL,AntSQL 或 all)")
    
    # 评测参数
    parser.add_argument("--etype", type=str, default="all", choices=['all', 'exec', 'match'], help="评测类型")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getenv("BILINGUAL_SQL_CODER_EVAL_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "evaluation_outputs")),
        help="输出目录"
    )
    
    args = parser.parse_args()
    
    # 解析要测试的数据集
    if args.datasets.lower() == 'all':
        datasets_to_test = list(DATASET_CONFIGS.keys())
    else:
        datasets_to_test = [ds.strip() for ds in args.datasets.split(',')]
    
    # 验证数据集名称
    invalid_datasets = [ds for ds in datasets_to_test if ds not in DATASET_CONFIGS]
    if invalid_datasets:
        print(f"❌ 错误：无效的数据集名称: {', '.join(invalid_datasets)}")
        print(f"支持的数据集: {', '.join(DATASET_CONFIGS.keys())}")
        sys.exit(1)
    
    # 生成模型标识符，用于区分不同模型的输出文件
    if args.model_type == 'base':
        model_identifier = 'base'
    else:
        # 从checkpoint路径提取模型标识
        if args.checkpoint_dir:
            # 提取checkpoint目录名作为标识（如 checkpoint-2700）
            ckpt_name = os.path.basename(args.checkpoint_dir.rstrip('/'))
            model_identifier = f"tuned_{ckpt_name}"
        else:
            model_identifier = 'tuned'
    
    print(f"\n{'='*80}")
    print(f"📊 Text2SQL 多数据集评测系统")
    print(f"{'='*80}")
    print(f"将要测试的数据集: {', '.join(datasets_to_test)}")
    print(f"模型类型: {args.model_type}")
    print(f"基座模型: {args.base_model_id}")
    print(f"模型标识: {model_identifier}")
    if args.model_type == 'tuned':
        print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"{'='*80}\n")
    
    # 初始化JSON数据库构建器
    db_builder = None
    if JSONDatabaseBuilder is not None:
        # 检查是否有需要构建数据库的数据集
        needs_building = any(DATASET_CONFIGS[ds].get('needs_db_building', False) 
                            for ds in datasets_to_test if ds in DATASET_CONFIGS)
        if needs_building:
            db_builder = JSONDatabaseBuilder(temp_dir=os.path.join(args.output_dir, "temp_databases"))
            print(f"✅ JSON数据库构建器已初始化")
    
    # 存储评测结果
    results = {}
    
    # 循环处理每个数据集
    for dataset_name in datasets_to_test:
        config = DATASET_CONFIGS[dataset_name]
        
        print(f"\n{'='*80}")
        emoji = "🔵" if config['language'] == 'en' else "🟢"
        print(f"{emoji} 处理 {dataset_name} 数据集 ({config['language'].upper()})")
        print(f"{'='*80}")
        
        # 检查文件是否存在
        if not os.path.exists(config['dev_file']):
            print(f"⚠️  警告: 开发集文件不存在: {config['dev_file']}")
            print(f"   跳过 {dataset_name} 数据集\n")
            continue
        
        # 如果需要生成gold SQL文件
        if config.get('needs_gold_generation', False):
            # Gold SQL生成到evaluation目录以避免权限问题
            gold_sql_path = os.path.join(args.output_dir, f"gold_{dataset_name.lower()}.sql")
            config['gold_sql'] = gold_sql_path  # 更新配置
            
            if not os.path.exists(gold_sql_path):
                print(f"\n💡 {dataset_name} 需要生成gold SQL文件...")
                try:
                    generation_type = config.get('gold_generation_type', '')
                    if dataset_name == 'Chase':
                        generate_gold_sql_for_chase(config['dev_file'], gold_sql_path)
                    elif generation_type in ['wikisql', 'antsql']:
                        generate_gold_sql_from_structured(
                            config['dev_file'],
                            config['tables_file'],
                            gold_sql_path,
                            generation_type
                        )
                    else:
                        print(f"⚠️  警告：未知的生成类型: {generation_type}")
                except Exception as e:
                    print(f"❌ 生成gold SQL失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"✅ Gold SQL文件已存在: {gold_sql_path}")
        
        # 如果需要从JSON构建数据库
        if config.get('needs_db_building', False) and db_builder is not None:
            db_type = config.get('db_type', '')
            print(f"\n🔨 {dataset_name} 需要从JSON构建数据库...")
            
            try:
                if db_type == 'dusql':
                    db_dir = db_builder.build_dusql_database(
                        db_schema_file=config['db_schema_file'],
                        db_content_file=config['db_content_file']
                    )
                    config['db_dir'] = db_dir  # 更新配置
                    print(f"✅ DuSQL数据库已构建: {db_dir}")
                elif db_type == 'chase':
                    db_dir = db_builder.build_chase_database(
                        tables_file=config['tables_file']
                    )
                    config['db_dir'] = db_dir  # 更新配置
                    print(f"✅ Chase数据库已构建: {db_dir}")
            except Exception as e:
                print(f"❌ 构建数据库失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 输出文件路径（包含模型标识，避免覆盖）
        output_file = os.path.join(args.output_dir, f"pred_{dataset_name.lower()}_{model_identifier}.sql")
        
        # 1. 推理阶段
        if not args.skip_inference:
            try:
                num_samples = run_inference(
                    model_type=args.model_type,
                    checkpoint_dir=args.checkpoint_dir,
                    output_file=output_file,
                    dataset_name=dataset_name,
                    dataset_config=config,
                    base_model_id=args.base_model_id
                )
                print(f"✅ {dataset_name} 推理完成，生成 {num_samples} 条SQL")
            except Exception as e:
                print(f"❌ {dataset_name} 推理失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        else:
            print(f"跳过推理，直接使用文件: {output_file}")
            if not os.path.exists(output_file):
                print(f"❌ 错误: 文件 {output_file} 不存在，无法进行评测。")
                continue
        
        # 2. 评测阶段（仅对有评测脚本的数据集）
        if config.get('has_evaluator', False):
            # 检查gold SQL文件是否存在
            gold_file = config.get('gold_sql')
            use_gold_file = gold_file and os.path.exists(gold_file)
            
            if not use_gold_file:
                if gold_file:
                    print(f"\n⚠️  警告: Gold SQL文件不存在: {gold_file}")
                print(f"⚠️  将跳过Exact Match评测，只计算Execution Accuracy")
                gold_file = None  # 设置为None以跳过exact match
            
            try:
                result = run_evaluation(
                    gold_file=gold_file,
                    pred_file=output_file,
                    db_dir=config['db_dir'],
                    tables_file=config['tables_file'],
                    etype=args.etype,
                    dataset_name=dataset_name
                )
                results[dataset_name] = result
            except Exception as e:
                print(f"❌ {dataset_name} 评测失败: {e}")
                import traceback
                traceback.print_exc()
                results[dataset_name] = None
        else:
            print(f"\n💡 {dataset_name} 暂无官方评测脚本，已生成预测SQL: {output_file}")
            results[dataset_name] = {"status": "inference_only", "output_file": output_file}
    
    # ========== 汇总结果 ==========
    print("\n" + "="*80)
    print("📈 评测结果汇总")
    print("="*80)
    
    if len(results) > 0:
        # 分类统计
        evaluated_datasets = {}  # 有评测分数的数据集
        inference_only_datasets = []  # 只做了推理的数据集
        
        for dataset_name, result in results.items():
            if result and isinstance(result, dict):
                if result.get('status') == 'inference_only':
                    inference_only_datasets.append(dataset_name)
                elif 'exact_match' in result or 'execution_accuracy' in result:
                    evaluated_datasets[dataset_name] = result
        
        # 显示有评测结果的数据集
        if evaluated_datasets:
            print("\n✅ 已完成评测的数据集:\n")
            for dataset_name, result in evaluated_datasets.items():
                emoji = "🔵" if DATASET_CONFIGS[dataset_name]['language'] == 'en' else "🟢"
                print(f"{emoji} {dataset_name} 数据集:")
                
                if 'exact_match' in result and result['exact_match'] is not None:
                    print(f"  Exact Match:        {result['exact_match']:.4f} ({result['exact_match']*100:.2f}%)")
                if 'execution_accuracy' in result and result['execution_accuracy'] is not None:
                    print(f"  Execution Accuracy: {result['execution_accuracy']:.4f} ({result['execution_accuracy']*100:.2f}%)")
                
                # 显示难度级别
                has_difficulty = False
                for level in ['easy', 'medium', 'hard', 'extra']:
                    if level in result and result[level] is not None:
                        if not has_difficulty:
                            print("  按难度分布:")
                            has_difficulty = True
                        print(f"    {level.capitalize():6s}: {result[level]:.4f} ({result[level]*100:.2f}%)")
                print()
        
        # 计算平均分（仅针对有评测结果的数据集）
        if len(evaluated_datasets) >= 2:
            print("\n⭐ 平均得分 (所有已评测数据集):")
            
            # 计算 Exact Match 平均
            em_scores = [res['exact_match'] for res in evaluated_datasets.values() if 'exact_match' in res and res['exact_match'] is not None]
            if em_scores:
                avg_em = sum(em_scores) / len(em_scores)
                print(f"  Exact Match:        {avg_em:.4f} ({avg_em*100:.2f}%)")
            
            # 计算 Execution Accuracy 平均
            exec_scores = [res['execution_accuracy'] for res in evaluated_datasets.values() if 'execution_accuracy' in res and res['execution_accuracy'] is not None]
            if exec_scores:
                avg_exec = sum(exec_scores) / len(exec_scores)
                print(f"  Execution Accuracy: {avg_exec:.4f} ({avg_exec*100:.2f}%)")
            
            # 计算各难度级别的平均分
            for level in ['easy', 'medium', 'hard', 'extra']:
                level_scores = [res[level] for res in evaluated_datasets.values() if level in res and res[level] is not None]
                if level_scores:
                    if level == 'easy':
                        print("  按难度分布:")
                    avg_level = sum(level_scores) / len(level_scores)
                    print(f"    {level.capitalize():6s}: {avg_level:.4f} ({avg_level*100:.2f}%)")
        
        # 显示只做了推理的数据集
        if inference_only_datasets:
            print("\n💡 以下数据集已生成预测SQL（暂无评测脚本）:")
            for dataset_name in inference_only_datasets:
                output_file = results[dataset_name].get('output_file', '')
                print(f"  • {dataset_name}: {output_file}")
        
        # 保存结果到文件（包含模型标识，避免覆盖）
        results_file = os.path.join(args.output_dir, f"evaluation_summary_{model_identifier}.json")
        detailed_results_file = os.path.join(args.output_dir, f"evaluation_detailed_{model_identifier}.txt")
        
        # 保存JSON格式的结果（只保存有评测分数的结果）
        with open(results_file, 'w', encoding='utf-8') as f:
            save_results = {k: v for k, v in evaluated_datasets.items()}
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 评测结果已保存到: {results_file}")
        
        # 保存详细的文本格式结果
        with open(detailed_results_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("📈 Text2SQL 多数据集评测结果\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"测试数据集: {', '.join(datasets_to_test)}\n")
            f.write(f"模型类型: {args.model_type}\n")
            if args.model_type == 'tuned':
                f.write(f"Checkpoint: {args.checkpoint_dir}\n")
            f.write("\n" + "="*80 + "\n\n")
            
            # 详细结果
            if evaluated_datasets:
                f.write("✅ 已完成评测的数据集:\n\n")
                for dataset_name, result in evaluated_datasets.items():
                    emoji = "🔵" if DATASET_CONFIGS[dataset_name]['language'] == 'en' else "🟢"
                    f.write(f"{emoji} {dataset_name} 数据集:\n")
                    
                    if 'exact_match' in result and result['exact_match'] is not None:
                        f.write(f"  Exact Match:        {result['exact_match']:.4f} ({result['exact_match']*100:.2f}%)\n")
                    else:
                        f.write(f"  Exact Match:        N/A (无gold SQL文件)\n")
                    
                    if 'execution_accuracy' in result and result['execution_accuracy'] is not None:
                        f.write(f"  Execution Accuracy: {result['execution_accuracy']:.4f} ({result['execution_accuracy']*100:.2f}%)\n")
                    else:
                        f.write(f"  Execution Accuracy: N/A (无数据库文件)\n")
                    
                    # 显示难度级别
                    has_difficulty = False
                    for level in ['easy', 'medium', 'hard', 'extra']:
                        if level in result and result[level] is not None:
                            if not has_difficulty:
                                f.write("  按难度分布:\n")
                                has_difficulty = True
                            f.write(f"    {level.capitalize():6s}: {result[level]:.4f} ({result[level]*100:.2f}%)\n")
                    f.write("\n")
                
                # 平均分
                if len(evaluated_datasets) >= 2:
                    f.write("\n" + "="*80 + "\n")
                    f.write("⭐ 平均得分 (所有已评测数据集):\n\n")
                    
                    # 计算 Exact Match 平均
                    em_scores = [res['exact_match'] for res in evaluated_datasets.values() if 'exact_match' in res and res['exact_match'] is not None]
                    if em_scores:
                        avg_em = sum(em_scores) / len(em_scores)
                        f.write(f"  Exact Match:        {avg_em:.4f} ({avg_em*100:.2f}%)\n")
                    
                    # 计算 Execution Accuracy 平均
                    exec_scores = [res['execution_accuracy'] for res in evaluated_datasets.values() if 'execution_accuracy' in res and res['execution_accuracy'] is not None]
                    if exec_scores:
                        avg_exec = sum(exec_scores) / len(exec_scores)
                        f.write(f"  Execution Accuracy: {avg_exec:.4f} ({avg_exec*100:.2f}%)\n")
                    
                    # 计算各难度级别的平均分
                    for level in ['easy', 'medium', 'hard', 'extra']:
                        level_scores = [res[level] for res in evaluated_datasets.values() if level in res and res[level] is not None]
                        if level_scores:
                            if level == 'easy':
                                f.write("  按难度分布:\n")
                            avg_level = sum(level_scores) / len(level_scores)
                            f.write(f"    {level.capitalize():6s}: {avg_level:.4f} ({avg_level*100:.2f}%)\n")
            
            # 只做了推理的数据集
            if inference_only_datasets:
                f.write("\n" + "="*80 + "\n")
                f.write("💡 以下数据集已生成预测SQL（暂无评测脚本）:\n")
                for dataset_name in inference_only_datasets:
                    output_file = results[dataset_name].get('output_file', '')
                    f.write(f"  • {dataset_name}: {output_file}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("✅ 全流程结束！\n")
        
        print(f"💾 详细结果已保存到: {detailed_results_file}")
    
    # 清理数据库构建器
    if db_builder is not None:
        # 不自动清理，保留数据库以便后续使用
        print(f"\n💾 临时数据库保留在: {db_builder.temp_dir}")
        print(f"   如需清理，请手动删除该目录")
    
    print("\n✅ 全流程结束！")

if __name__ == "__main__":
    main()
