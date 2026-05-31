"""
通用Text2SQL评测脚本
支持所有数据集的评测，包括文本匹配和执行评测
改进版：完全集成Spider官方评测器的逻辑，使用相同的指标计算方法
"""

import os
import sys
import json
import sqlite3
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# 尝试导入sqlparse用于更好的SQL标准化
try:
    import sqlparse
    HAS_SQLPARSE = True
except ImportError:
    HAS_SQLPARSE = False
    print("⚠️  提示: 未安装sqlparse，将使用基础标准化方法。建议安装: pip install sqlparse")

# 尝试导入Spider官方评测器
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPIDER_EVAL_DIR = os.getenv(
    "SPIDER_EVAL_DIR",
    os.path.join(PROJECT_ROOT, "data_collected", "spider", "eval")
)
HAS_SPIDER_EVAL = False
try:
    if os.path.exists(SPIDER_EVAL_DIR):
        sys.path.insert(0, SPIDER_EVAL_DIR)
        from process_sql import get_schema, get_sql, Schema
        from evaluation import (
            Evaluator as SpiderEvaluator,
            build_foreign_key_map_from_json,
            rebuild_sql_val,
            rebuild_sql_col,
            build_valid_col_units,
            eval_exec_match
        )
        HAS_SPIDER_EVAL = True
        print("✅ 成功加载Spider官方评测器")
except ImportError as e:
    print(f"⚠️  警告: 无法加载Spider官方评测器: {e}")
    print("   将使用通用评测方法")


class UniversalEvaluator:
    """
    通用评测器，支持多种数据集格式
    对于Spider/CSpider数据集，使用官方评测器的精确匹配算法
    对于其他数据集，使用改进的字符串标准化方法
    """
    
    def __init__(self, dataset_name: str = "Unknown", use_spider_official: bool = None):
        self.dataset_name = dataset_name
        # 自动判断是否使用Spider官方评测器
        if use_spider_official is None:
            # Spider和CSpider使用官方评测器
            self.use_spider_official = HAS_SPIDER_EVAL and dataset_name in ['Spider', 'CSpider']
        else:
            self.use_spider_official = use_spider_official and HAS_SPIDER_EVAL
        
        self.reset_scores()
        
        # 初始化外键映射字典（避免None访问）
        self.kmaps = {}
        
        # 如果使用Spider官方评测器，初始化相关资源
        if self.use_spider_official:
            self.spider_evaluator = SpiderEvaluator()
            print(f"📊 {dataset_name} 使用Spider官方评测方法")
        else:
            print(f"📊 {dataset_name} 使用通用评测方法")
    
    def load_kmaps(self, tables_file: str):
        """加载外键映射（Spider官方评测器需要）"""
        if self.use_spider_official and tables_file and os.path.exists(tables_file):
            try:
                self.kmaps = build_foreign_key_map_from_json(tables_file)
                print(f"✅ 已加载外键映射: {len(self.kmaps)} 个数据库")
            except Exception as e:
                print(f"⚠️  警告: 加载外键映射失败: {e}")
                self.kmaps = {}
    
    def reset_scores(self):
        """重置评分"""
        self.scores = {
            'exact_match': 0,
            'execution_match': 0,
            'total': 0,
            'valid_exec': 0,  # 有效的执行评测数量
            'details': []
        }
    
    def normalize_sql(self, sql: str) -> str:
        """
        标准化SQL语句以便比较
        参考Spider官方评测器的标准化策略，使评测更加宽松
        """
        if not sql:
            return ""
        
        # 1. 基础清理
        sql = sql.strip()
        
        # 2. 移除SQL注释
        # 移除单行注释 --
        sql = re.sub(r'--[^\n]*', '', sql)
        # 移除多行注释 /* */
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        
        # 3. 如果安装了sqlparse，使用它进行标准化
        if HAS_SQLPARSE:
            try:
                # sqlparse会自动处理很多标准化问题
                parsed = sqlparse.parse(sql)
                if parsed:
                    sql = str(parsed[0])
                    # 格式化：统一关键字、去除多余空格
                    sql = sqlparse.format(
                        sql,
                        keyword_case='upper',
                        identifier_case='lower',
                        strip_comments=True,
                        reindent=False,
                        use_space_around_operators=True
                    )
            except:
                pass  # 如果解析失败，继续使用原始SQL
        
        # 4. 转小写（用于最终比较）
        sql = sql.lower()
        
        # 5. 标准化空格和换行
        # 将所有连续空白字符替换为单个空格
        sql = re.sub(r'\s+', ' ', sql)
        
        # 6. 移除末尾分号
        sql = sql.rstrip(';').strip()
        
        # 7. 标准化引号：双引号转单引号
        # 但要小心处理嵌套引号的情况
        sql = re.sub(r'"([^"]*)"', r"'\1'", sql)
        
        # 8. 标准化操作符周围的空格
        # 确保操作符两边都有空格（或都没有），这里选择都有空格
        operators = ['=', '!=', '<>', '>', '<', '>=', '<=', '+', '-', '*', '/', '%']
        for op in operators:
            # 移除操作符周围的空格，然后添加单个空格
            sql = re.sub(r'\s*' + re.escape(op) + r'\s*', f' {op} ', sql)
        
        # 9. 标准化逗号后的空格
        sql = re.sub(r',\s*', ', ', sql)
        
        # 10. 标准化括号周围的空格
        # 左括号前加空格，右括号后加空格（某些情况除外）
        sql = re.sub(r'\s*\(\s*', ' ( ', sql)
        sql = re.sub(r'\s*\)\s*', ' ) ', sql)
        
        # 11. 标准化 DISTINCT 关键字
        sql = re.sub(r'\bdistinct\s+', 'distinct ', sql, flags=re.IGNORECASE)
        
        # 12. 标准化 AS 别名
        # 移除 AS 关键字，因为在SQL中 AS 是可选的
        sql = re.sub(r'\s+as\s+', ' ', sql, flags=re.IGNORECASE)
        
        # 13. 标准化表名和列名引用
        # 移除不必要的反引号、方括号
        sql = sql.replace('`', '')
        sql = re.sub(r'\[([^\]]+)\]', r'\1', sql)
        
        # 14. 标准化字符串中的空格（保持字符串内容不变）
        # 这一步比较复杂，暂时跳过，依赖execution accuracy来验证
        
        # 15. 最终清理：移除首尾空格，合并多余空格
        sql = ' '.join(sql.split())
        
        # 16. 标准化特殊SQL关键字的间隔
        # ORDER BY, GROUP BY 等
        sql = re.sub(r'\border\s+by\b', 'order by', sql)
        sql = re.sub(r'\bgroup\s+by\b', 'group by', sql)
        sql = re.sub(r'\bhaving\s+', 'having ', sql)
        sql = re.sub(r'\bwhere\s+', 'where ', sql)
        sql = re.sub(r'\blimit\s+', 'limit ', sql)
        sql = re.sub(r'\boffset\s+', 'offset ', sql)
        
        # 17. 移除多余的括号（小心处理）
        # 例如 ((expr)) -> (expr)
        # 但这可能改变语义，所以暂时不做
        
        return sql.strip()
    
    def exact_match_score(self, pred_sql: str, gold_sql: str) -> bool:
        """
        精确匹配评分（改进版）
        对于Spider数据集使用官方评测器的语法树比较
        对于其他数据集使用改进的字符串标准化比较
        """
        pred_norm = self.normalize_sql(pred_sql)
        gold_norm = self.normalize_sql(gold_sql)
        
        # 直接比较标准化后的SQL
        if pred_norm == gold_norm:
            return True
        
        # 额外的宽松匹配策略
        # 1. 尝试移除所有空格后比较（处理极端空格差异）
        pred_no_space = pred_norm.replace(' ', '')
        gold_no_space = gold_norm.replace(' ', '')
        if pred_no_space == gold_no_space:
            return True
        
        # 2. 尝试排序SELECT子句中的列（如果是简单SELECT）
        # 例如: SELECT a, b 和 SELECT b, a 应该被认为等价（某些情况下）
        # 但这可能改变语义，所以需要谨慎
        # 暂时不实现，依赖execution accuracy
        
        return False
    
    def spider_exact_match_score(self, pred_sql: str, gold_sql: str, db_path: str, db_id: str) -> bool:
        """
        使用Spider官方评测器的精确匹配方法
        通过SQL解析器将SQL转换为AST后进行结构化比较
        """
        if not HAS_SPIDER_EVAL or not db_path or not os.path.exists(db_path):
            # 如果无法使用官方评测器，回退到字符串匹配
            return self.exact_match_score(pred_sql, gold_sql)
        
        try:
            # 1. 加载数据库schema
            schema = Schema(get_schema(db_path))
            
            # 2. 解析gold SQL为AST
            try:
                gold_parsed = get_sql(schema, gold_sql)
            except Exception as e:
                print(f"⚠️  警告: Gold SQL解析失败: {e}")
                return False
            
            # 3. 解析pred SQL为AST
            try:
                pred_parsed = get_sql(schema, pred_sql)
            except Exception as e:
                # 预测SQL解析失败，使用空SQL结构
                pred_parsed = {
                    "except": None,
                    "from": {"conds": [], "table_units": []},
                    "groupBy": [],
                    "having": [],
                    "intersect": None,
                    "limit": None,
                    "orderBy": [],
                    "select": [False, []],
                    "union": None,
                    "where": []
                }
            
            # 4. 标准化SQL结构（禁用值比较，禁用DISTINCT比较）
            # 这是Spider官方评测器的关键：比较SQL结构而非具体值
            kmap = self.kmaps.get(db_id, {}) if self.kmaps else {}
            
            # 重建gold SQL（移除值，标准化列引用）
            g_valid_col_units = build_valid_col_units(gold_parsed['from']['table_units'], schema)
            gold_parsed = rebuild_sql_val(gold_parsed)
            gold_parsed = rebuild_sql_col(g_valid_col_units, gold_parsed, kmap)
            
            # 重建pred SQL
            p_valid_col_units = build_valid_col_units(pred_parsed['from']['table_units'], schema)
            pred_parsed = rebuild_sql_val(pred_parsed)
            pred_parsed = rebuild_sql_col(p_valid_col_units, pred_parsed, kmap)
            
            # 5. 使用Spider评测器进行精确匹配
            exact_match = self.spider_evaluator.eval_exact_match(pred_parsed, gold_parsed)
            
            return exact_match == 1
            
        except Exception as e:
            print(f"⚠️  Spider评测失败: {e}，回退到字符串匹配")
            return self.exact_match_score(pred_sql, gold_sql)
    
    def spider_execution_match_score(self, pred_sql: str, gold_sql: str, db_path: str, db_id: str) -> Optional[bool]:
        """
        使用Spider官方评测器的执行匹配方法
        """
        if not HAS_SPIDER_EVAL or not db_path or not os.path.exists(db_path):
            return self.execution_match_score(pred_sql, gold_sql, db_path)
        
        try:
            # 加载schema
            schema = Schema(get_schema(db_path))
            
            # 解析SQL
            try:
                gold_parsed = get_sql(schema, gold_sql)
            except:
                return None
            
            try:
                pred_parsed = get_sql(schema, pred_sql)
            except:
                return False
            
            # 使用官方的执行匹配评测
            result = eval_exec_match(db_path, pred_sql, gold_sql, pred_parsed, gold_parsed)
            return result == 1
            
        except Exception as e:
            # 回退到通用方法
            return self.execution_match_score(pred_sql, gold_sql, db_path)
    
    def execution_match_score(self, pred_sql: str, gold_sql: str, db_path: str) -> Optional[bool]:
        """
        执行匹配评分（改进版）
        参考Spider官方评测器，使用更宽松的结果比较策略
        """
        if not db_path or not os.path.exists(db_path):
            return None
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 执行预测SQL
            try:
                cursor.execute(pred_sql)
                pred_results = cursor.fetchall()
            except Exception as e:
                conn.close()
                return False
            
            # 执行金标准SQL
            try:
                cursor.execute(gold_sql)
                gold_results = cursor.fetchall()
            except Exception as e:
                conn.close()
                return None  # 金标准SQL执行失败，不计入评测
            
            conn.close()
            
            # 比较结果
            # 1. 如果长度不同，肯定不匹配
            if len(pred_results) != len(gold_results):
                return False
            
            # 2. 如果都是空结果，匹配
            if len(pred_results) == 0:
                return True
            
            # 3. 转换为集合比较（处理行顺序问题）
            try:
                # 将每一行转换为tuple，然后放入set
                pred_set = set()
                for row in pred_results:
                    if isinstance(row, (list, tuple)):
                        # 标准化行中的值：None统一处理，浮点数四舍五入
                        normalized_row = []
                        for val in row:
                            if val is None:
                                normalized_row.append(None)
                            elif isinstance(val, float):
                                # 浮点数精度问题：保留6位小数
                                normalized_row.append(round(val, 6))
                            elif isinstance(val, str):
                                # 字符串去除首尾空格
                                normalized_row.append(val.strip())
                            else:
                                normalized_row.append(val)
                        pred_set.add(tuple(normalized_row))
                    else:
                        pred_set.add((row,))
                
                gold_set = set()
                for row in gold_results:
                    if isinstance(row, (list, tuple)):
                        normalized_row = []
                        for val in row:
                            if val is None:
                                normalized_row.append(None)
                            elif isinstance(val, float):
                                normalized_row.append(round(val, 6))
                            elif isinstance(val, str):
                                normalized_row.append(val.strip())
                            else:
                                normalized_row.append(val)
                        gold_set.add(tuple(normalized_row))
                    else:
                        gold_set.add((row,))
                
                return pred_set == gold_set
            except TypeError:
                # 如果无法转换为集合（比如包含不可哈希的类型，如list），尝试直接比较
                # 但要考虑顺序问题：排序后比较
                try:
                    pred_sorted = sorted([tuple(row) if isinstance(row, list) else row for row in pred_results])
                    gold_sorted = sorted([tuple(row) if isinstance(row, list) else row for row in gold_results])
                    return pred_sorted == gold_sorted
                except:
                    # 最后的兜底：直接比较
                    return pred_results == gold_results
                
        except Exception as e:
            return None
    
    def evaluate_single(self, pred_sql: str, gold_sql: str, db_path: Optional[str] = None, db_id: str = "") -> Dict:
        """评估单个样本"""
        result = {
            'exact_match': 0,
            'execution_match': None,
            'pred_sql': pred_sql,
            'gold_sql': gold_sql
        }
        
        # 精确匹配
        if self.use_spider_official and db_path and db_id:
            # 使用Spider官方评测器
            if self.spider_exact_match_score(pred_sql, gold_sql, db_path, db_id):
                result['exact_match'] = 1
        else:
            # 使用通用评测器
            if self.exact_match_score(pred_sql, gold_sql):
                result['exact_match'] = 1
        
        # 执行匹配（如果有数据库）
        if db_path:
            if self.use_spider_official and db_id:
                exec_result = self.spider_execution_match_score(pred_sql, gold_sql, db_path, db_id)
            else:
                exec_result = self.execution_match_score(pred_sql, gold_sql, db_path)
            
            if exec_result is not None:
                result['execution_match'] = 1 if exec_result else 0
        
        return result
    
    def load_gold_sql(self, gold_file: str) -> List[Tuple[str, str]]:
        """
        加载gold SQL文件
        支持多种格式：
        1. Spider格式: SQL\tdb_id
        2. DuSQL格式: qid\tSQL\tdb_id
        3. 纯SQL文件: 每行一个SQL
        """
        gold_data = []
        
        with open(gold_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                
                if len(parts) >= 2:
                    # 判断是Spider格式还是DuSQL格式
                    if parts[0].startswith('qid'):  # DuSQL格式
                        if len(parts) >= 3:
                            sql, db_id = parts[1], parts[2]
                        else:
                            sql, db_id = parts[1], ""
                    else:  # Spider格式
                        sql, db_id = parts[0], parts[1] if len(parts) > 1 else ""
                else:
                    # 纯SQL格式
                    sql, db_id = parts[0], ""
                
                gold_data.append((sql, db_id))
        
        return gold_data
    
    def load_pred_sql(self, pred_file: str) -> List[str]:
        """加载预测SQL文件，每行一个SQL"""
        pred_data = []
        
        with open(pred_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                pred_data.append(line if line else "")
        
        return pred_data
    
    def evaluate(self, pred_file: str, gold_file: Optional[str] = None, db_dir: Optional[str] = None, tables_file: Optional[str] = None) -> Dict:
        """
        完整评测流程
        
        Args:
            pred_file: 预测SQL文件路径
            gold_file: 金标准SQL文件路径（可选，如果不提供则只计算execution accuracy）
            db_dir: 数据库目录（可选）
            tables_file: 表结构文件（Spider官方评测器需要，用于加载外键映射）
        
        Returns:
            评测结果字典
        """
        self.reset_scores()
        
        # 如果使用Spider官方评测器，加载外键映射
        if self.use_spider_official and tables_file:
            self.load_kmaps(tables_file)
        
        # 加载数据
        gold_data = None
        if gold_file and os.path.exists(gold_file):
            gold_data = self.load_gold_sql(gold_file)
        
        pred_data = self.load_pred_sql(pred_file)
        
        # 如果没有gold数据，只能做execution评测
        if gold_data is None:
            print(f"⚠️  注意: 未提供gold SQL文件，只进行execution评测")
            # 创建虚拟gold_data用于遍历
            gold_data = [("", "") for _ in pred_data]
        
        # 确保数量一致
        if len(pred_data) != len(gold_data):
            print(f"⚠️  警告: 预测数量({len(pred_data)})与金标准数量({len(gold_data)})不一致")
            min_len = min(len(pred_data), len(gold_data))
            pred_data = pred_data[:min_len]
            gold_data = gold_data[:min_len]
        
        # 逐个评测
        for i, ((gold_sql, db_id), pred_sql) in enumerate(zip(gold_data, pred_data)):
            # 构建数据库路径
            db_path = None
            if db_dir and db_id:
                # 尝试多种可能的数据库文件位置
                possible_paths = [
                    os.path.join(db_dir, db_id, f"{db_id}.sqlite"),
                    os.path.join(db_dir, f"{db_id}.sqlite"),
                    os.path.join(db_dir, db_id, f"{db_id}.db"),
                    os.path.join(db_dir, f"{db_id}.db"),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        db_path = path
                        break
            
            # 评测单个样本（如果没有gold_sql则跳过exact match）
            if gold_sql:  # 只有当有gold SQL时才计算exact match
                result = self.evaluate_single(pred_sql, gold_sql, db_path, db_id)
            else:  # 只计算execution match
                result = {
                    'exact_match': None,
                    'execution_match': None,
                    'pred_sql': pred_sql,
                    'gold_sql': ''
                }
                # 尝试执行预测SQL（不需要gold SQL）
                if db_path and pred_sql:
                    try:
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        cursor.execute(pred_sql)
                        cursor.fetchall()
                        conn.close()
                        result['execution_match'] = 1  # 能成功执行
                    except:
                        result['execution_match'] = 0  # 执行失败
            
            # 累计分数
            self.scores['total'] += 1
            if result['exact_match'] is not None:
                self.scores['exact_match'] += result['exact_match']
            
            if result['execution_match'] is not None:
                self.scores['valid_exec'] += 1
                self.scores['execution_match'] += result['execution_match']
            
            # 保存详情（可选，用于调试）
            if result['exact_match'] is not None and result['exact_match'] == 0:  # 只保存错误的
                self.scores['details'].append({
                    'index': i,
                    'db_id': db_id,
                    'exact_match': result['exact_match'],
                    'execution_match': result['execution_match'],
                    'pred': pred_sql[:100],  # 截断以节省空间
                    'gold': gold_sql[:100] if gold_sql else ''
                })
        
        # 计算最终分数
        total = self.scores['total']
        results = {
            'dataset': self.dataset_name,
            'total_samples': total,
        }
        
        # 只有当计算了exact match时才添加该指标
        if gold_file and os.path.exists(gold_file):
            results['exact_match'] = self.scores['exact_match'] / total if total > 0 else 0
            results['exact_match_count'] = self.scores['exact_match']
        else:
            results['exact_match'] = None
            results['exact_match_count'] = 0
        
        # 添加执行准确率（如果有）
        if self.scores['valid_exec'] > 0:
            results['execution_accuracy'] = self.scores['execution_match'] / self.scores['valid_exec']
            results['execution_match_count'] = self.scores['execution_match']
            results['valid_exec_count'] = self.scores['valid_exec']
        else:
            results['execution_accuracy'] = None
        
        return results
    
    def print_results(self, results: Dict):
        """打印评测结果"""
        print(f"\n{'='*60}")
        print(f"评测结果 - {results['dataset']}")
        print(f"{'='*60}")
        print(f"总样本数: {results['total_samples']}")
        
        if results['exact_match'] is not None:
            print(f"Exact Match: {results['exact_match']:.4f} ({results['exact_match_count']}/{results['total_samples']})")
        else:
            print(f"Exact Match: N/A (无gold SQL文件)")
        
        if results['execution_accuracy'] is not None:
            print(f"Execution Accuracy: {results['execution_accuracy']:.4f} ({results['execution_match_count']}/{results['valid_exec_count']})")
        else:
            print(f"Execution Accuracy: N/A (无数据库文件)")
        
        print(f"{'='*60}\n")


def evaluate_dataset(pred_file: str, gold_file: str, db_dir: Optional[str] = None, 
                     dataset_name: str = "Unknown") -> Dict:
    """
    便捷函数：评测单个数据集
    
    Args:
        pred_file: 预测SQL文件
        gold_file: 金标准SQL文件
        db_dir: 数据库目录（可选）
        dataset_name: 数据集名称
    
    Returns:
        评测结果字典
    """
    evaluator = UniversalEvaluator(dataset_name)
    results = evaluator.evaluate(pred_file, gold_file, db_dir)
    evaluator.print_results(results)
    return results


def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="通用Text2SQL评测脚本")
    parser.add_argument("--pred", type=str, required=True, help="预测SQL文件路径")
    parser.add_argument("--gold", type=str, required=True, help="金标准SQL文件路径")
    parser.add_argument("--db", type=str, default=None, help="数据库目录（可选）")
    parser.add_argument("--dataset", type=str, default="Unknown", help="数据集名称")
    
    args = parser.parse_args()
    
    results = evaluate_dataset(
        pred_file=args.pred,
        gold_file=args.gold,
        db_dir=args.db,
        dataset_name=args.dataset
    )
    
    return results


if __name__ == "__main__":
    main()
