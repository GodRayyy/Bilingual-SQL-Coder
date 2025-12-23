#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结构化SQL转标准SQL转换器
支持WikiSQL和AntSQL的结构化SQL格式转换为标准SQL字符串

WikiSQL格式: {"sel": int, "agg": int, "conds": [[col_idx, op, value], ...]}
AntSQL格式: {"sel": [col_idx], "agg": [agg_type], "conds": [[col_idx, op, value], ...], ...}
"""

import json
import os
from typing import Dict, List, Any, Optional


class StructuredSQLConverter:
    """结构化SQL转标准SQL转换器"""
    
    # WikiSQL聚合函数映射
    WIKISQL_AGG_OPS = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    
    # WikiSQL条件操作符映射
    WIKISQL_COND_OPS = ['=', '>', '<', 'OP']
    
    # AntSQL聚合函数映射
    ANTSQL_AGG_OPS = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    
    # AntSQL条件操作符映射
    ANTSQL_COND_OPS = ['>', '<', '==', '!=', 'LIKE']
    
    def __init__(self):
        """初始化转换器"""
        self.wikisql_table_cache = {}  # 缓存WikiSQL的表信息
        self.antsql_columns = []  # AntSQL的列信息
    
    def load_wikisql_tables(self, tables_file: str) -> Dict[str, Dict]:
        """
        加载WikiSQL的表结构信息
        
        Args:
            tables_file: tables.jsonl文件路径
            
        Returns:
            表ID到表信息的映射字典
        """
        print(f"📚 加载WikiSQL表结构: {tables_file}")
        tables = {}
        
        with open(tables_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    table = json.loads(line)
                    table_id = table.get('id', table.get('table_id'))
                    tables[table_id] = {
                        'header': table.get('header', []),
                        'types': table.get('types', []),
                        'name': table.get('name', table_id)
                    }
        
        self.wikisql_table_cache = tables
        print(f"✅ 已加载 {len(tables)} 个WikiSQL表")
        return tables
    
    def load_antsql_columns(self, tables_file: str) -> List[str]:
        """
        加载AntSQL的列信息
        
        Args:
            tables_file: Excel文件路径
            
        Returns:
            列名列表
        """
        print(f"📚 加载AntSQL列信息: {tables_file}")
        
        try:
            import pandas as pd
            df = pd.read_excel(tables_file)
            columns = df.columns.tolist()
            self.antsql_columns = columns
            print(f"✅ 已加载 {len(columns)} 个AntSQL列")
            return columns
        except Exception as e:
            print(f"⚠️  警告: 加载AntSQL列失败: {e}")
            return []
    
    def wikisql_to_sql(self, sql_dict: Dict[str, Any], table_id: str) -> str:
        """
        将WikiSQL格式转换为标准SQL
        
        Args:
            sql_dict: WikiSQL的SQL字典 {"sel": int, "agg": int, "conds": [...]}
            table_id: 表ID
            
        Returns:
            标准SQL字符串
        """
        if table_id not in self.wikisql_table_cache:
            print(f"⚠️  警告: 表 {table_id} 不在缓存中")
            return "SELECT * FROM table"
        
        table_info = self.wikisql_table_cache[table_id]
        headers = table_info['header']
        
        # 处理SELECT子句
        sel_idx = sql_dict.get('sel', 0)
        agg_idx = sql_dict.get('agg', 0)
        
        if sel_idx >= len(headers):
            print(f"⚠️  警告: 列索引 {sel_idx} 超出范围")
            select_col = '*'
        else:
            select_col = f'`{headers[sel_idx]}`'
        
        # 添加聚合函数
        if agg_idx > 0 and agg_idx < len(self.WIKISQL_AGG_OPS):
            agg_func = self.WIKISQL_AGG_OPS[agg_idx]
            select_clause = f"{agg_func}({select_col})"
        else:
            select_clause = select_col
        
        # 处理FROM子句
        table_name = table_info.get('name', table_id)
        from_clause = f"`{table_name}`"
        
        # 处理WHERE子句
        conds = sql_dict.get('conds', [])
        where_clauses = []
        
        for cond in conds:
            if len(cond) != 3:
                continue
            
            col_idx, op_idx, value = cond
            
            if col_idx >= len(headers):
                continue
            
            col_name = f'`{headers[col_idx]}`'
            
            # 获取操作符
            if op_idx < len(self.WIKISQL_COND_OPS):
                op = self.WIKISQL_COND_OPS[op_idx]
            else:
                op = '='
            
            # 处理值（字符串需要加引号）
            if isinstance(value, str):
                value_str = f"'{value}'"
            else:
                value_str = str(value)
            
            where_clauses.append(f"{col_name} {op} {value_str}")
        
        # 组装SQL
        sql = f"SELECT {select_clause} FROM {from_clause}"
        
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        
        return sql
    
    def antsql_to_sql(self, sql_dict: Dict[str, Any]) -> str:
        """
        将AntSQL格式转换为标准SQL
        
        Args:
            sql_dict: AntSQL的SQL字典
            
        Returns:
            标准SQL字符串
        """
        # 处理SELECT子句
        sel_list = sql_dict.get('sel', [0])
        agg_list = sql_dict.get('agg', [0])
        
        if not self.antsql_columns:
            print("⚠️  警告: AntSQL列信息未加载")
            return "SELECT * FROM FundTable"
        
        select_parts = []
        for i, sel_idx in enumerate(sel_list):
            if sel_idx >= len(self.antsql_columns):
                continue
            
            col_name = f'`{self.antsql_columns[sel_idx]}`'
            
            # 添加聚合函数
            if i < len(agg_list) and agg_list[i] > 0 and agg_list[i] < len(self.ANTSQL_AGG_OPS):
                agg_func = self.ANTSQL_AGG_OPS[agg_list[i]]
                select_parts.append(f"{agg_func}({col_name})")
            else:
                select_parts.append(col_name)
        
        if not select_parts:
            select_parts = ['*']
        
        select_clause = ", ".join(select_parts)
        
        # 处理FROM子句
        from_clause = "`FundTable`"
        
        # 处理WHERE子句
        conds = sql_dict.get('conds', [])
        where_clauses = []
        
        for cond in conds:
            if len(cond) != 3:
                continue
            
            col_idx, op_idx, value = cond
            
            if col_idx >= len(self.antsql_columns):
                continue
            
            col_name = f'`{self.antsql_columns[col_idx]}`'
            
            # 获取操作符
            if op_idx < len(self.ANTSQL_COND_OPS):
                op = self.ANTSQL_COND_OPS[op_idx]
                # 将 == 转换为 =
                if op == '==':
                    op = '='
            else:
                op = '='
            
            # 处理值（字符串需要加引号）
            if isinstance(value, str):
                value_str = f"'{value}'"
            else:
                value_str = str(value)
            
            where_clauses.append(f"{col_name} {op} {value_str}")
        
        # 处理连接操作符
        cond_conn_op = sql_dict.get('cond_conn_op', 0)
        conn_op = " AND " if cond_conn_op == 0 else " OR "
        
        # 组装SQL
        sql = f"SELECT {select_clause} FROM {from_clause}"
        
        if where_clauses:
            sql += " WHERE " + conn_op.join(where_clauses)
        
        # 处理ORDER BY
        orderby = sql_dict.get('orderby', [])
        if orderby:
            order_parts = []
            asc_desc = sql_dict.get('asc_desc', 0)
            direction = "ASC" if asc_desc == 0 else "DESC"
            
            for col_idx in orderby:
                if col_idx < len(self.antsql_columns):
                    order_parts.append(f"`{self.antsql_columns[col_idx]}`")
            
            if order_parts:
                sql += f" ORDER BY {', '.join(order_parts)} {direction}"
        
        # 处理LIMIT
        limit = sql_dict.get('limit', 0)
        if limit > 0:
            sql += f" LIMIT {limit}"
        
        return sql
    
    def convert_wikisql_file(self, dev_file: str, tables_file: str, output_file: str) -> int:
        """
        转换WikiSQL的dev文件
        
        Args:
            dev_file: dev.jsonl文件路径
            tables_file: tables.jsonl文件路径
            output_file: 输出的SQL文件路径
            
        Returns:
            转换的SQL数量
        """
        print(f"\n🔄 开始转换WikiSQL...")
        print(f"输入文件: {dev_file}")
        print(f"输出文件: {output_file}")
        
        # 加载表信息
        self.load_wikisql_tables(tables_file)
        
        # 转换SQL
        converted_sqls = []
        with open(dev_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    item = json.loads(line)
                    table_id = item.get('table_id', '')
                    sql_dict = item.get('sql', {})
                    
                    sql = self.wikisql_to_sql(sql_dict, table_id)
                    converted_sqls.append(sql)
                    
                except Exception as e:
                    print(f"⚠️  警告: 第 {line_num} 行转换失败: {e}")
                    converted_sqls.append("SELECT * FROM table")
        
        # 保存结果
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for sql in converted_sqls:
                f.write(sql + '\n')
        
        print(f"✅ WikiSQL转换完成: {len(converted_sqls)} 条SQL")
        return len(converted_sqls)
    
    def convert_antsql_file(self, dev_file: str, tables_file: str, output_file: str) -> int:
        """
        转换AntSQL的dev文件
        
        Args:
            dev_file: antsql1_dev.jsonl文件路径
            tables_file: Excel表格路径
            output_file: 输出的SQL文件路径
            
        Returns:
            转换的SQL数量
        """
        print(f"\n🔄 开始转换AntSQL...")
        print(f"输入文件: {dev_file}")
        print(f"输出文件: {output_file}")
        
        # 加载列信息
        self.load_antsql_columns(tables_file)
        
        # 转换SQL
        converted_sqls = []
        with open(dev_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    item = json.loads(line)
                    sql_dict = item.get('sql', {})
                    
                    sql = self.antsql_to_sql(sql_dict)
                    converted_sqls.append(sql)
                    
                except Exception as e:
                    print(f"⚠️  警告: 第 {line_num} 行转换失败: {e}")
                    converted_sqls.append("SELECT * FROM FundTable")
        
        # 保存结果
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for sql in converted_sqls:
                f.write(sql + '\n')
        
        print(f"✅ AntSQL转换完成: {len(converted_sqls)} 条SQL")
        return len(converted_sqls)


def test_converter():
    """测试转换器"""
    converter = StructuredSQLConverter()
    
    # 测试WikiSQL转换
    print("\n" + "="*80)
    print("测试WikiSQL转换")
    print("="*80)
    
    wikisql_dev = "/data0/dywang/Llm/Text2Sql/data_collected/WikiSQL/data/dev.jsonl"
    wikisql_tables = "/data0/dywang/Llm/Text2Sql/data_collected/WikiSQL/data/dev.tables.jsonl"
    wikisql_output = "/data0/tygao/classes/text2sql/evaluation/test_wikisql_gold.sql"
    
    if os.path.exists(wikisql_dev) and os.path.exists(wikisql_tables):
        converter.convert_wikisql_file(wikisql_dev, wikisql_tables, wikisql_output)
        
        # 显示前5条转换结果
        print("\n前5条转换结果:")
        with open(wikisql_output, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(f"  {i+1}. {line.strip()}")
    else:
        print("⚠️  WikiSQL文件不存在，跳过测试")
    
    # 测试AntSQL转换
    print("\n" + "="*80)
    print("测试AntSQL转换")
    print("="*80)
    
    antsql_dev = "/data0/dywang/Llm/Text2Sql/data_collected/antsql1/antsql1_dev.jsonl"
    antsql_tables = "/data0/dywang/Llm/Text2Sql/data_collected/antsql1/antsql1_fundTable.xlsx"
    antsql_output = "/data0/tygao/classes/text2sql/evaluation/test_antsql_gold.sql"
    
    if os.path.exists(antsql_dev) and os.path.exists(antsql_tables):
        converter.convert_antsql_file(antsql_dev, antsql_tables, antsql_output)
        
        # 显示前5条转换结果
        print("\n前5条转换结果:")
        with open(antsql_output, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(f"  {i+1}. {line.strip()}")
    else:
        print("⚠️  AntSQL文件不存在，跳过测试")


if __name__ == "__main__":
    test_converter()
