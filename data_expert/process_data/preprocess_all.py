import json
import os
import random
from tqdm import tqdm
import pandas as pd # 仅用于依赖检查

# ================= 配置区域 =================

DATA_SOURCE_DIR = "/amax/home/dywang/Llm/Text2Sql/data_collected"
CURRENT_WORK_DIR = os.path.abspath(".")
OUTPUT_DIR = os.path.join(CURRENT_WORK_DIR, "processed_output_messages")

os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPTS = {
    "en": "You are a professional SQL data analyst. Given a database schema and a question, write a valid SQL query.",
    "zh": "你是一位专业的SQL数据分析师。给定数据库Schema和自然语言问题，请生成一个有效的SQL查询。"
}

DATASET_CONFIG = {
    # --- 英文数据集 ---
    "Spider": {
        "type": "spider",
        "language": "en", 
        "train_data": os.path.join("spider", "train_spider.json"),
        "dev_data": os.path.join("spider", "dev.json"),
        "tables": os.path.join("spider", "tables.json")
    },
    "Bird": {
        "type": "spider",
        "language": "en",
        "train_data": os.path.join("Bird", "train", "train.json"),
        "dev_data": os.path.join("Bird", "dev", "dev.json"),
        "tables": os.path.join("Bird", "train", "train_tables.json"),
        "tables_dev": os.path.join("Bird", "dev", "dev_tables.json") 
    },
    "WikiSQL": {
        "type": "wikisql",
        "language": "en",
        "train_data": os.path.join("WikiSQL", "data", "train.jsonl"),
        "dev_data": os.path.join("WikiSQL", "data", "dev.jsonl"),
        "tables": os.path.join("WikiSQL", "data", "train.tables.jsonl"),
        "tables_dev": os.path.join("WikiSQL", "data", "dev.tables.jsonl")
    },
    # --- 中文数据集 ---
    "Chase": {
        "type": "spider", # Schema 格式依然是 Spider 风格
        "language": "zh", 
        "train_data": os.path.join("chase", "data", "train.json"),
        "dev_data": os.path.join("chase", "data", "dev.json"),
        "tables": os.path.join("chase", "data", "tables.json")
    },
    "CSpider": {
        "type": "spider",
        "language": "zh", 
        "train_data": os.path.join("CSpider", "train.json"),
        "dev_data": os.path.join("CSpider", "dev.json"),
        "tables": os.path.join("CSpider", "tables.json")
    },
    "DuSQL": {
        "type": "spider",
        "language": "zh", 
        "train_data": os.path.join("DuSQL", "train.json"),
        "dev_data": os.path.join("DuSQL", "dev.json"),
        "tables": os.path.join("DuSQL", "db_schema.json")
    },
    "AntSQL": {
        "type": "antsql",
        "language": "zh", 
        "train_data": os.path.join("antsql1", "antsql1_train.jsonl"),
        "dev_data": os.path.join("antsql1", "antsql1_dev.jsonl"),
        "tables": os.path.join("antsql1", "antsql1_fundTable.xlsx")
    }
}

# ================= Schema 加载 (保持不变) =================

def load_spider_style_schema(tables_paths):
    if isinstance(tables_paths, str): tables_paths = [tables_paths]
    schema_map = {}
    for path in tables_paths:
        if not path or not os.path.exists(path): continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for db in data:
                db_id = db.get('db_id', db.get('database_id')) 
                if not db_id: continue

                table_names = db.get('table_names_original', db.get('table_names'))
                column_names = db.get('column_names_original', db.get('column_names'))
                
                if not table_names or not column_names: continue

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
        except Exception as e:
            print(f"Error loading schema {path}: {e}")
    return schema_map

def load_wikisql_style_schema(tables_paths):
    if isinstance(tables_paths, str): tables_paths = [tables_paths]
    schema_map = {}
    for path in tables_paths:
        if not path or not os.path.exists(path): continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                schema_map[item['id']] = f"Table w_{item['id']}, columns = [{', '.join(item['header'])}]"
    return schema_map

def load_antsql_schema(excel_path):
    schema_map = {}
    if not os.path.exists(excel_path): return {}
    try:
        df = pd.read_excel(excel_path)
        cols = df.columns.tolist()
        col_str = ", ".join([str(c) for c in cols])
        schema_map['antsql_default'] = f"Table fund_table, columns = [{col_str}]"
    except Exception as e:
        print(f"Error reading AntSQL excel: {e}")
    return schema_map

# ================= 辅助函数 =================

def sanitize_sql(raw_val):
    if raw_val is None: return ""
    if isinstance(raw_val, (dict, list)):
        return json.dumps(raw_val, ensure_ascii=False)
    val_str = str(raw_val).strip()
    return val_str

def get_sql_field(item):
    # 增加 utterance (Chase) 的支持通常不需要在这里，因为 Chase 结构特殊，单独处理
    keys = ['query', 'sql', 'SQL', 'final_sql', 'formatted_sql']
    for k in keys:
        if k in item and item[k] is not None:
            return item[k]
    return None

def get_question_field(item):
    keys = ['question', 'question_text', 'input', 'utterance'] # 增加 utterance
    for k in keys:
        if k in item and item[k]:
            return item[k]
    return ""

def get_db_id(item):
    keys = ['db_id', 'database_id', 'db']
    for k in keys:
        if k in item:
            return item[k]
    return ""

# ================= 数据收集逻辑 (重要修改) =================

def collect_dataset(name, config):
    ds_type = config['type']
    lang = config.get('language', 'en')
    system_prompt = PROMPTS[lang]
    
    # 1. 加载 Schema
    schema_map = {}
    if ds_type == 'spider':
        paths = [config['tables']]
        if 'tables_dev' in config: paths.append(config['tables_dev'])
        schema_map = load_spider_style_schema(paths)
    elif ds_type == 'wikisql':
        paths = [config['tables']]
        if 'tables_dev' in config: paths.append(config['tables_dev'])
        schema_map = load_wikisql_style_schema(paths)
    elif ds_type == 'antsql':
        schema_map = load_antsql_schema(config['tables'])

    collected_data = {'train': [], 'dev': []}
    stats = {'total': 0, 'skipped_no_sql': 0, 'success': 0}

    for split in ['train', 'dev']:
        data_path = config.get(f'{split}_data')
        if not data_path or not os.path.exists(data_path):
            continue
            
        print(f"Collecting {name} {split} ({lang})...")
        
        items = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                if data_path.endswith('.jsonl'):
                    for line in f:
                        if line.strip(): items.append(json.loads(line))
                else:
                    items = json.load(f)
        except Exception as e:
            print(f"!!! Error reading file {data_path}: {e}")
            continue

        for item in tqdm(items, desc=f"{name} {split}", leave=False):
            stats['total'] += 1
            
            # === 分支 1: 处理 Chase 类型的多轮对话 (Interaction) ===
            if 'interaction' in item and isinstance(item['interaction'], list):
                # 获取 DB Schema
                db_id = get_db_id(item)
                schema_str = schema_map.get(db_id, "")
                
                # 构建多轮 Messages
                messages = [{"role": "system", "content": system_prompt}]
                
                valid_interaction = False
                for idx, turn in enumerate(item['interaction']):
                    # Chase 里的问题叫 utterance，SQL 叫 query
                    question = turn.get('utterance', '')
                    sql_val = turn.get('query', '')
                    
                    final_sql = sanitize_sql(sql_val)
                    if not final_sql or not question:
                        continue
                        
                    valid_interaction = True
                    
                    # 第一轮：带上 Schema
                    # 后续轮次：不带 Schema (依赖历史上下文)
                    if idx == 0:
                        if lang == 'zh':
                            user_content = f"数据库Schema:\n{schema_str}\n\n问题: {question}\n\nSQL:"
                        else:
                            user_content = f"Database Schema:\n{schema_str}\n\nQuestion: {question}\n\nSQL:"
                    else:
                        if lang == 'zh':
                            user_content = f"问题: {question}\n\nSQL:"
                        else:
                            user_content = f"Question: {question}\n\nSQL:"
                    
                    messages.append({"role": "user", "content": user_content})
                    messages.append({"role": "assistant", "content": final_sql})
                
                # 只有当 interaction 中至少有一组有效问答时才保存
                if valid_interaction:
                    entry = {"messages": messages, "source": name}
                    collected_data[split].append(entry)
                    stats['success'] += 1
                else:
                    stats['skipped_no_sql'] += 1

            # === 分支 2: 处理 Spider/Bird/WikiSQL 等单轮任务 ===
            else:
                question = ""
                sql_val = ""
                schema_str = ""
                
                if ds_type == 'spider': # Bird 也走这里
                    question = get_question_field(item)
                    raw_sql = get_sql_field(item)
                    sql_val = raw_sql
                    db_id = get_db_id(item)
                    schema_str = schema_map.get(db_id, "")
                    
                elif ds_type == 'wikisql':
                    question = item.get('question', '')
                    sql_val = item.get('query')
                    if sql_val is None and 'sql' in item: 
                        sql_val = json.dumps(item['sql'], ensure_ascii=False)
                    schema_str = schema_map.get(item.get('table_id'), "")

                elif ds_type == 'antsql':
                    question = item.get('question', '')
                    sql_val = get_sql_field(item)
                    schema_str = schema_map.get('antsql_default', "")

                final_sql = sanitize_sql(sql_val)
                
                if not final_sql:
                    stats['skipped_no_sql'] += 1
                    continue

                # 构造单轮 Messages
                if lang == 'zh':
                    user_content = f"数据库Schema:\n{schema_str}\n\n问题: {question}\n\nSQL:"
                else:
                    user_content = f"Database Schema:\n{schema_str}\n\nQuestion: {question}\n\nSQL:"
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": final_sql}
                ]

                entry = {"messages": messages, "source": name}
                collected_data[split].append(entry)
                stats['success'] += 1
    
    if stats['total'] > 0:
        print(f"   -> {name} Result: {stats['success']} parsed, {stats['skipped_no_sql']} skipped (Total items processed: {stats['total']})")
    
    return collected_data['train'], collected_data['dev']

# ================= 主函数 =================

def main():
    train_out_path = os.path.join(OUTPUT_DIR, "text2sql_train_messages.jsonl")
    dev_out_path = os.path.join(OUTPUT_DIR, "text2sql_dev_messages.jsonl")
    
    all_train = []
    all_dev = []
    
    print(f"Data Source Directory: {DATA_SOURCE_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    if not os.path.exists(DATA_SOURCE_DIR):
        print(f"Error: Data source directory does not exist: {DATA_SOURCE_DIR}")
        return

    for name, config in DATASET_CONFIG.items():
        dataset_subdir = config['train_data'].split(os.sep)[0]
        real_path = os.path.join(DATA_SOURCE_DIR, dataset_subdir)
        
        if not os.path.exists(real_path):
            print(f"Skipping {name} (Directory not found: {real_path})")
            continue
            
        full_config = config.copy()
        for k in ['train_data', 'dev_data', 'tables', 'tables_dev']:
            if k in full_config:
                full_config[k] = os.path.join(DATA_SOURCE_DIR, full_config[k])
        
        t_data, d_data = collect_dataset(name, full_config)
        all_train.extend(t_data)
        all_dev.extend(d_data)

    print(f"\nTotal Train Examples: {len(all_train)}")
    print(f"Total Dev Examples:   {len(all_dev)}")

    print("Shuffling data...")
    random.seed(42)
    random.shuffle(all_train)
    random.shuffle(all_dev)

    def write_jsonl(path, data):
        print(f"Saving to {path}...")
        with open(path, 'w', encoding='utf-8') as f:
            for entry in data:
                line = json.dumps(entry, ensure_ascii=False)
                f.write(line + '\n')

    write_jsonl(train_out_path, all_train)
    write_jsonl(dev_out_path, all_dev)

    print(f"\nDone! Data saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()