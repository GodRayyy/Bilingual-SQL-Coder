import sqlite3
import pandas as pd
import os

def get_all_databases(root_path, mode='folder'):
    """
    扫描目录下所有的数据库
    mode='folder': 寻找 root/db_name/db_name.sqlite
    mode='file':   寻找 root/db_name.sqlite
    """
    dbs = []
    if not os.path.exists(root_path):
        return []
    
    try:
        if mode == 'folder':
            # 遍历子目录结构 (Spider/Bird/CSpider)
            for dirname in os.listdir(root_path):
                dir_path = os.path.join(root_path, dirname)
                if os.path.isdir(dir_path):
                    # 检查里面是否有对应的 sqlite 文件
                    db_file = os.path.join(dir_path, f"{dirname}.sqlite")
                    if os.path.exists(db_file):
                        dbs.append(dirname)
                        
        elif mode == 'file':
            # 遍历文件结构 (DuSQL)
            for filename in os.listdir(root_path):
                if filename.endswith(".sqlite"):
                    # 去掉后缀作为数据库名
                    db_name = filename.replace(".sqlite", "")
                    dbs.append(db_name)
    except Exception as e:
        print(f"Error scanning databases: {e}")

    return sorted(dbs)

def get_db_path(root_path, db_name, mode='folder'):
    """获取数据库文件的绝对路径"""
    if mode == 'folder':
        return os.path.join(root_path, db_name, f"{db_name}.sqlite")
    else:
        return os.path.join(root_path, f"{db_name}.sqlite")

def get_db_connection(root_path, db_name, mode='folder'):
    """获取数据库连接"""
    db_path = get_db_path(root_path, db_name, mode)
    return sqlite3.connect(db_path)

def get_db_schema(root_path, db_name, mode='folder'):
    """
    提取数据库 Schema
    """
    try:
        conn = get_db_connection(root_path, db_name, mode)
        cursor = conn.cursor()
        
        schema_str = ""
        
        # 获取所有表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}';")
            result = cursor.fetchone()
            if result:
                create_sql = result[0]
                schema_str += f"{create_sql}\n"
            
        conn.close()
        return schema_str
    except Exception as e:
        return f"Schema extraction failed: {str(e)}"

def execute_sql(root_path, db_name, sql, mode='folder'):
    """执行生成的 SQL 并返回 DataFrame"""
    try:
        conn = get_db_connection(root_path, db_name, mode)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)