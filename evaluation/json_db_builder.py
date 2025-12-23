#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON数据库构建工具
从JSON格式的数据库内容创建临时SQLite数据库，用于执行评测
"""

import os
import json
import sqlite3
import tempfile
import shutil
from typing import Dict, List, Optional


class JSONDatabaseBuilder:
    """从JSON文件构建SQLite数据库"""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        初始化数据库构建器
        
        Args:
            temp_dir: 临时目录路径，如果不提供则自动创建
        """
        if temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="text2sql_db_")
            self.auto_cleanup = True
        else:
            self.temp_dir = temp_dir
            os.makedirs(temp_dir, exist_ok=True)
            self.auto_cleanup = False
        
        print(f"✅ 临时数据库目录: {self.temp_dir}")
    
    def cleanup(self):
        """清理临时目录"""
        if self.auto_cleanup and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"🧹 已清理临时目录: {self.temp_dir}")
    
    def build_dusql_database(self, db_schema_file: str, db_content_file: str) -> str:
        """
        为DuSQL数据集构建SQLite数据库
        
        Args:
            db_schema_file: db_schema.json文件路径
            db_content_file: db_content.json文件路径
        
        Returns:
            数据库目录路径
        """
        print("\n🔨 开始构建DuSQL数据库...")
        
        # 加载schema和content
        with open(db_schema_file, 'r', encoding='utf-8') as f:
            schemas = json.load(f)
        
        with open(db_content_file, 'r', encoding='utf-8') as f:
            contents = json.load(f)
        
        # 创建schema索引
        schema_map = {item['db_id']: item for item in schemas}
        content_map = {item['db_id']: item for item in contents}
        
        print(f"📊 找到 {len(schema_map)} 个数据库")
        
        # 为每个数据库创建SQLite文件
        db_dir = os.path.join(self.temp_dir, "dusql_databases")
        os.makedirs(db_dir, exist_ok=True)
        
        created_count = 0
        for db_id, schema in schema_map.items():
            try:
                db_path = os.path.join(db_dir, f"{db_id}.sqlite")
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # 获取表名和列名
                table_names = schema.get('table_names', [])
                column_names = schema.get('column_names', [])
                
                # 为每个表创建SQL
                for table_idx, table_name in enumerate(table_names):
                    # 获取该表的所有列
                    columns = []
                    for col_idx, col_name in enumerate(column_names):
                        if isinstance(col_name, list) and len(col_name) >= 2:
                            if col_name[0] == table_idx:
                                # 清理列名（移除特殊字符）
                                clean_col = col_name[1].replace(' ', '_')
                                columns.append(f'"{clean_col}" TEXT')
                    
                    if columns:
                        create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(columns)})'
                        cursor.execute(create_sql)
                
                # 插入数据（如果有）
                if db_id in content_map:
                    content = content_map[db_id]
                    tables_data = content.get('tables', {})
                    
                    for table_name, table_data in tables_data.items():
                        if table_name in table_names:
                            cells = table_data.get('cell', [])
                            if cells:
                                # 获取列数
                                table_idx = table_names.index(table_name)
                                columns = [col_name[1] for col_name in column_names 
                                          if isinstance(col_name, list) and len(col_name) >= 2 
                                          and col_name[0] == table_idx]
                                
                                if columns:
                                    # 插入每一行
                                    placeholders = ', '.join(['?' for _ in columns])
                                    col_names = ', '.join([f'"{col}"' for col in columns])
                                    insert_sql = f'INSERT INTO "{table_name}" ({col_names}) VALUES ({placeholders})'
                                    
                                    for row in cells:
                                        try:
                                            # 确保行数据长度与列数一致
                                            if len(row) == len(columns):
                                                cursor.execute(insert_sql, row)
                                        except Exception as e:
                                            # 跳过有问题的行
                                            pass
                
                conn.commit()
                conn.close()
                created_count += 1
                
            except Exception as e:
                print(f"  ⚠️  创建数据库 {db_id} 失败: {e}")
                continue
        
        print(f"✅ 成功创建 {created_count}/{len(schema_map)} 个DuSQL数据库")
        return db_dir
    
    def build_chase_database(self, tables_file: str) -> str:
        """
        为Chase数据集构建SQLite数据库
        
        Args:
            tables_file: tables.json文件路径
        
        Returns:
            数据库目录路径
        """
        print("\n🔨 开始构建Chase数据库...")
        
        # SQLite保留的系统表名（不能被用户创建）
        SQLITE_RESERVED_TABLES = {
            'sqlite_sequence',
            'sqlite_master',
            'sqlite_temp_master',
            'sqlite_stat1',
            'sqlite_stat2',
            'sqlite_stat3',
            'sqlite_stat4'
        }
        
        # 加载tables信息
        with open(tables_file, 'r', encoding='utf-8') as f:
            tables_data = json.load(f)
        
        print(f"📊 找到 {len(tables_data)} 个数据库schema")
        
        # 为每个数据库创建SQLite文件
        db_dir = os.path.join(self.temp_dir, "chase_databases")
        os.makedirs(db_dir, exist_ok=True)
        
        created_count = 0
        skipped_reserved_tables = 0
        
        for db_info in tables_data:
            try:
                db_id = db_info.get('db_id')
                if not db_id:
                    continue
                
                db_path = os.path.join(db_dir, f"{db_id}.sqlite")
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # 获取表名和列名
                table_names = db_info.get('table_names_original', db_info.get('table_names', []))
                column_names = db_info.get('column_names_original', db_info.get('column_names', []))
                
                # 为每个表创建SQL（只创建结构，不插入数据）
                for table_idx, table_name in enumerate(table_names):
                    # 跳过SQLite保留的系统表名
                    if table_name.lower() in SQLITE_RESERVED_TABLES:
                        skipped_reserved_tables += 1
                        continue
                    
                    # 获取该表的所有列
                    columns = []
                    for col_info in column_names:
                        if isinstance(col_info, list) and len(col_info) >= 2:
                            if col_info[0] == table_idx:
                                # 清理列名
                                clean_col = str(col_info[1]).replace(' ', '_')
                                columns.append(f'"{clean_col}" TEXT')
                    
                    if columns:
                        create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(columns)})'
                        cursor.execute(create_sql)
                
                conn.commit()
                conn.close()
                created_count += 1
                
            except Exception as e:
                print(f"  ⚠️  创建数据库 {db_id} 失败: {e}")
                continue
        
        print(f"✅ 成功创建 {created_count}/{len(tables_data)} 个Chase数据库（空表结构）")
        if skipped_reserved_tables > 0:
            print(f"💡 跳过了 {skipped_reserved_tables} 个SQLite保留系统表")
        print("⚠️  注意: Chase数据库只包含表结构，没有实际数据，execution评测可能不准确")
        return db_dir
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.auto_cleanup:
            self.cleanup()


def test_database_creation():
    """测试数据库创建"""
    base_dir = "/data0/dywang/Llm/Text2Sql/data_collected"
    
    with JSONDatabaseBuilder() as builder:
        # 测试DuSQL
        print("\n" + "="*80)
        print("测试DuSQL数据库创建")
        print("="*80)
        dusql_db_dir = builder.build_dusql_database(
            db_schema_file=os.path.join(base_dir, "DuSQL/db_schema.json"),
            db_content_file=os.path.join(base_dir, "DuSQL/db_content.json")
        )
        
        # 验证创建的数据库
        db_files = [f for f in os.listdir(dusql_db_dir) if f.endswith('.sqlite')]
        print(f"\n✅ 创建的数据库文件: {len(db_files)} 个")
        if db_files:
            sample_db = os.path.join(dusql_db_dir, db_files[0])
            conn = sqlite3.connect(sample_db)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"   示例数据库 {db_files[0]} 包含表: {[t[0] for t in tables]}")
            conn.close()
        
        # 测试Chase
        print("\n" + "="*80)
        print("测试Chase数据库创建")
        print("="*80)
        chase_db_dir = builder.build_chase_database(
            tables_file=os.path.join(base_dir, "chase/data/tables.json")
        )
        
        # 验证创建的数据库
        db_files = [f for f in os.listdir(chase_db_dir) if f.endswith('.sqlite')]
        print(f"\n✅ 创建的数据库文件: {len(db_files)} 个")


if __name__ == "__main__":
    test_database_creation()
