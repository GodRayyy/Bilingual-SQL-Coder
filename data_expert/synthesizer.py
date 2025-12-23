"""
双语数据处理专家 - 数据合成生成模块
Data Synthesizer for Text-to-SQL datasets
"""

import json
import logging
import random
from typing import List, Dict, Any, Optional

from .api_client import QwenClient
from .prompts import (
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_TEMPLATE
)
from .config import DEFAULT_MODEL, COMPLEX_MODEL

logger = logging.getLogger(__name__)


class DataSynthesizer:
    """Text-to-SQL数据合成器"""
    
    # 预定义的领域和Schema模板
    DOMAIN_TEMPLATES = {
        "企业销售": {
            "db_id": "enterprise_sales",
            "schema": {
                "tables": [
                    {
                        "name": "products",
                        "name_zh": "产品表",
                        "columns": [
                            {"name": "product_id", "name_zh": "产品ID", "type": "INTEGER PRIMARY KEY"},
                            {"name": "product_name", "name_zh": "产品名称", "type": "TEXT"},
                            {"name": "category", "name_zh": "类别", "type": "TEXT"},
                            {"name": "price", "name_zh": "单价", "type": "REAL"},
                            {"name": "stock", "name_zh": "库存", "type": "INTEGER"}
                        ]
                    },
                    {
                        "name": "orders",
                        "name_zh": "订单表",
                        "columns": [
                            {"name": "order_id", "name_zh": "订单ID", "type": "INTEGER PRIMARY KEY"},
                            {"name": "customer_id", "name_zh": "客户ID", "type": "INTEGER"},
                            {"name": "order_date", "name_zh": "订单日期", "type": "DATE"},
                            {"name": "total_amount", "name_zh": "订单金额", "type": "REAL"},
                            {"name": "status", "name_zh": "状态", "type": "TEXT"}
                        ]
                    },
                    {
                        "name": "order_items",
                        "name_zh": "订单明细表",
                        "columns": [
                            {"name": "item_id", "name_zh": "明细ID", "type": "INTEGER PRIMARY KEY"},
                            {"name": "order_id", "name_zh": "订单ID", "type": "INTEGER"},
                            {"name": "product_id", "name_zh": "产品ID", "type": "INTEGER"},
                            {"name": "quantity", "name_zh": "数量", "type": "INTEGER"},
                            {"name": "unit_price", "name_zh": "单价", "type": "REAL"}
                        ]
                    },
                    {
                        "name": "customers",
                        "name_zh": "客户表",
                        "columns": [
                            {"name": "customer_id", "name_zh": "客户ID", "type": "INTEGER PRIMARY KEY"},
                            {"name": "customer_name", "name_zh": "客户名称", "type": "TEXT"},
                            {"name": "region", "name_zh": "地区", "type": "TEXT"},
                            {"name": "contact", "name_zh": "联系方式", "type": "TEXT"},
                            {"name": "level", "name_zh": "客户等级", "type": "TEXT"}
                        ]
                    }
                ]
            }
        },
        "学生成绩": {
            "db_id": "student_scores",
            "schema": {
                "tables": [
                    {
                        "name": "students",
                        "name_zh": "学生表",
                        "columns": [
                            {"name": "student_id", "name_zh": "学生ID", "type": "INTEGER PRIMARY KEY"},
                            {"name": "name", "name_zh": "姓名", "type": "TEXT"},
                            {"name": "gender", "name_zh": "性别", "type": "TEXT"},
                            {"name": "class_id", "name_zh": "班级ID", "type": "INTEGER"},
                            {"name": "enrollment_year", "name_zh": "入学年份", "type": "INTEGER"}
                        ]
                    },
                    {
                        "name": "courses",
                        "name_zh": "课程表",
                        "columns": [
                            {"name": "course_id", "name_zh": "课程ID", "type": "INTEGER PRIMARY KEY"},
                            {"name": "course_name", "name_zh": "课程名称", "type": "TEXT"},
                            {"name": "credits", "name_zh": "学分", "type": "INTEGER"},
                            {"name": "teacher_id", "name_zh": "教师ID", "type": "INTEGER"}
                        ]
                    },
                    {
                        "name": "scores",
                        "name_zh": "成绩表",
                        "columns": [
                            {"name": "score_id", "name_zh": "成绩ID", "type": "INTEGER PRIMARY KEY"},
                            {"name": "student_id", "name_zh": "学生ID", "type": "INTEGER"},
                            {"name": "course_id", "name_zh": "课程ID", "type": "INTEGER"},
                            {"name": "score", "name_zh": "分数", "type": "REAL"},
                            {"name": "semester", "name_zh": "学期", "type": "TEXT"}
                        ]
                    },
                    {
                        "name": "teachers",
                        "name_zh": "教师表",
                        "columns": [
                            {"name": "teacher_id", "name_zh": "教师ID", "type": "INTEGER PRIMARY KEY"},
                            {"name": "name", "name_zh": "姓名", "type": "TEXT"},
                            {"name": "department", "name_zh": "院系", "type": "TEXT"},
                            {"name": "title", "name_zh": "职称", "type": "TEXT"}
                        ]
                    }
                ]
            }
        },
        "医院管理": {
            "db_id": "hospital",
            "schema": {
                "tables": [
                    {
                        "name": "patients",
                        "name_zh": "患者表",
                        "columns": [
                            {"name": "patient_id", "name_zh": "患者ID", "type": "INTEGER PRIMARY KEY"},
                            {"name": "name", "name_zh": "姓名", "type": "TEXT"},
                            {"name": "age", "name_zh": "年龄", "type": "INTEGER"},
                            {"name": "gender", "name_zh": "性别", "type": "TEXT"},
                            {"name": "phone", "name_zh": "电话", "type": "TEXT"}
                        ]
                    },
                    {
                        "name": "doctors",
                        "name_zh": "医生表",
                        "columns": [
                            {"name": "doctor_id", "name_zh": "医生ID", "type": "INTEGER PRIMARY KEY"},
                            {"name": "name", "name_zh": "姓名", "type": "TEXT"},
                            {"name": "department", "name_zh": "科室", "type": "TEXT"},
                            {"name": "title", "name_zh": "职称", "type": "TEXT"}
                        ]
                    },
                    {
                        "name": "appointments",
                        "name_zh": "预约表",
                        "columns": [
                            {"name": "appointment_id", "name_zh": "预约ID", "type": "INTEGER PRIMARY KEY"},
                            {"name": "patient_id", "name_zh": "患者ID", "type": "INTEGER"},
                            {"name": "doctor_id", "name_zh": "医生ID", "type": "INTEGER"},
                            {"name": "appointment_date", "name_zh": "预约日期", "type": "DATE"},
                            {"name": "status", "name_zh": "状态", "type": "TEXT"}
                        ]
                    },
                    {
                        "name": "medical_records",
                        "name_zh": "病历表",
                        "columns": [
                            {"name": "record_id", "name_zh": "病历ID", "type": "INTEGER PRIMARY KEY"},
                            {"name": "patient_id", "name_zh": "患者ID", "type": "INTEGER"},
                            {"name": "doctor_id", "name_zh": "医生ID", "type": "INTEGER"},
                            {"name": "diagnosis", "name_zh": "诊断", "type": "TEXT"},
                            {"name": "prescription", "name_zh": "处方", "type": "TEXT"},
                            {"name": "visit_date", "name_zh": "就诊日期", "type": "DATE"}
                        ]
                    }
                ]
            }
        },
        "电商平台": {
            "db_id": "ecommerce",
            "schema": {
                "tables": [
                    {
                        "name": "users",
                        "name_zh": "用户表",
                        "columns": [
                            {"name": "user_id", "name_zh": "用户ID", "type": "INTEGER PRIMARY KEY"},
                            {"name": "username", "name_zh": "用户名", "type": "TEXT"},
                            {"name": "email", "name_zh": "邮箱", "type": "TEXT"},
                            {"name": "register_date", "name_zh": "注册日期", "type": "DATE"},
                            {"name": "vip_level", "name_zh": "VIP等级", "type": "INTEGER"}
                        ]
                    },
                    {
                        "name": "products",
                        "name_zh": "商品表",
                        "columns": [
                            {"name": "product_id", "name_zh": "商品ID", "type": "INTEGER PRIMARY KEY"},
                            {"name": "title", "name_zh": "商品标题", "type": "TEXT"},
                            {"name": "category", "name_zh": "类目", "type": "TEXT"},
                            {"name": "price", "name_zh": "价格", "type": "REAL"},
                            {"name": "seller_id", "name_zh": "卖家ID", "type": "INTEGER"},
                            {"name": "rating", "name_zh": "评分", "type": "REAL"}
                        ]
                    },
                    {
                        "name": "orders",
                        "name_zh": "订单表",
                        "columns": [
                            {"name": "order_id", "name_zh": "订单ID", "type": "INTEGER PRIMARY KEY"},
                            {"name": "user_id", "name_zh": "用户ID", "type": "INTEGER"},
                            {"name": "product_id", "name_zh": "商品ID", "type": "INTEGER"},
                            {"name": "quantity", "name_zh": "数量", "type": "INTEGER"},
                            {"name": "order_time", "name_zh": "下单时间", "type": "DATETIME"},
                            {"name": "payment_status", "name_zh": "支付状态", "type": "TEXT"}
                        ]
                    },
                    {
                        "name": "reviews",
                        "name_zh": "评价表",
                        "columns": [
                            {"name": "review_id", "name_zh": "评价ID", "type": "INTEGER PRIMARY KEY"},
                            {"name": "user_id", "name_zh": "用户ID", "type": "INTEGER"},
                            {"name": "product_id", "name_zh": "商品ID", "type": "INTEGER"},
                            {"name": "rating", "name_zh": "评分", "type": "INTEGER"},
                            {"name": "content", "name_zh": "评价内容", "type": "TEXT"},
                            {"name": "review_time", "name_zh": "评价时间", "type": "DATETIME"}
                        ]
                    }
                ]
            }
        }
    }
    
    def __init__(self, client: QwenClient = None):
        """
        初始化数据合成器
        
        Args:
            client: Qwen API客户端
        """
        self.client = client or QwenClient()
        self.model = DEFAULT_MODEL
        self.complex_model = COMPLEX_MODEL
    
    def synthesize_from_schema(
        self,
        schema: Dict[str, Any],
        db_id: str,
        domain: str,
        n_samples: int = 10,
        use_complex_model: bool = False
    ) -> List[Dict[str, Any]]:
        """
        基于Schema生成合成数据
        
        Args:
            schema: 数据库Schema
            db_id: 数据库ID
            domain: 领域描述
            n_samples: 生成样本数量
            use_complex_model: 是否使用复杂模型
            
        Returns:
            生成的样本列表
        """
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        
        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": SYNTHESIS_USER_TEMPLATE.format(
                schema=schema_str,
                db_id=db_id,
                domain=domain,
                n=n_samples
            )}
        ]
        
        model = self.complex_model if use_complex_model else self.model
        config = self.client.get_task_config("synthesis")
        result = self.client.call_with_json_output(messages, model=model, **config)
        
        if isinstance(result, dict):
            result = [result]
        
        # 添加元信息
        for item in result:
            item["db_id"] = db_id
            item["domain"] = domain
            item["synthetic"] = True
        
        return result
    
    def synthesize_from_domain(
        self,
        domain: str,
        n_samples: int = 10,
        use_complex_model: bool = False
    ) -> List[Dict[str, Any]]:
        """
        基于预定义领域生成合成数据
        
        Args:
            domain: 领域名称（如"企业销售"、"学生成绩"等）
            n_samples: 生成样本数量
            use_complex_model: 是否使用复杂模型
            
        Returns:
            生成的样本列表
        """
        if domain not in self.DOMAIN_TEMPLATES:
            available = list(self.DOMAIN_TEMPLATES.keys())
            raise ValueError(f"未知领域 '{domain}'，可用领域: {available}")
        
        template = self.DOMAIN_TEMPLATES[domain]
        
        return self.synthesize_from_schema(
            schema=template["schema"],
            db_id=template["db_id"],
            domain=domain,
            n_samples=n_samples,
            use_complex_model=use_complex_model
        )
    
    def synthesize_multi_domain(
        self,
        n_per_domain: int = 10,
        domains: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        多领域合成数据
        
        Args:
            n_per_domain: 每个领域生成的样本数
            domains: 指定领域列表，None则使用所有预定义领域
            
        Returns:
            所有领域的合成数据
        """
        domains = domains or list(self.DOMAIN_TEMPLATES.keys())
        all_results = []
        
        for domain in domains:
            try:
                logger.info(f"正在生成领域 '{domain}' 的数据...")
                results = self.synthesize_from_domain(domain, n_per_domain)
                all_results.extend(results)
                logger.info(f"领域 '{domain}' 生成了 {len(results)} 条数据")
            except Exception as e:
                logger.error(f"领域 '{domain}' 生成失败: {e}")
        
        return all_results
    
    def synthesize_by_difficulty(
        self,
        schema: Dict[str, Any],
        db_id: str,
        domain: str,
        difficulty: str,
        n_samples: int = 5
    ) -> List[Dict[str, Any]]:
        """
        按难度级别生成数据
        
        Args:
            schema: 数据库Schema
            db_id: 数据库ID
            domain: 领域描述
            difficulty: 难度级别 (easy/medium/hard)
            n_samples: 生成样本数量
            
        Returns:
            生成的样本列表
        """
        difficulty_prompts = {
            "easy": "只生成简单查询：单表SELECT、简单WHERE条件（1-2个条件）",
            "medium": "生成中等难度查询：多表JOIN、GROUP BY、聚合函数、3-4个条件",
            "hard": "生成困难查询：嵌套子查询、复杂聚合、多重JOIN、HAVING子句"
        }
        
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        
        prompt = f"""基于以下数据库Schema，生成{n_samples}条{difficulty}难度的Text-to-SQL样本：

数据库ID：{db_id}
领域：{domain}
Schema：
{schema_str}

难度要求：{difficulty_prompts.get(difficulty, difficulty_prompts['medium'])}

输出JSON列表：
[
  {{
    "question_zh": "中文问题",
    "question_en": "English question",
    "sql": "SELECT ...",
    "difficulty": "{difficulty}"
  }},
  ...
]

只输出JSON列表。"""

        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        model = self.complex_model if difficulty == "hard" else self.model
        config = self.client.get_task_config("synthesis")
        result = self.client.call_with_json_output(messages, model=model, **config)
        
        if isinstance(result, dict):
            result = [result]
        
        for item in result:
            item["db_id"] = db_id
            item["domain"] = domain
            item["synthetic"] = True
        
        return result
    
    def synthesize_balanced_dataset(
        self,
        schema: Dict[str, Any],
        db_id: str,
        domain: str,
        total_samples: int = 30
    ) -> List[Dict[str, Any]]:
        """
        生成难度均衡的数据集
        
        Args:
            schema: 数据库Schema
            db_id: 数据库ID
            domain: 领域描述
            total_samples: 总样本数
            
        Returns:
            均衡的样本列表
        """
        # 按40% easy, 40% medium, 20% hard分配
        n_easy = int(total_samples * 0.4)
        n_medium = int(total_samples * 0.4)
        n_hard = total_samples - n_easy - n_medium
        
        all_samples = []
        
        for difficulty, n in [("easy", n_easy), ("medium", n_medium), ("hard", n_hard)]:
            if n > 0:
                try:
                    samples = self.synthesize_by_difficulty(
                        schema, db_id, domain, difficulty, n
                    )
                    all_samples.extend(samples)
                except Exception as e:
                    logger.error(f"生成 {difficulty} 难度数据失败: {e}")
        
        # 打乱顺序
        random.shuffle(all_samples)
        
        return all_samples
    
    def get_available_domains(self) -> List[str]:
        """获取可用的预定义领域列表"""
        return list(self.DOMAIN_TEMPLATES.keys())
    
    def get_domain_schema(self, domain: str) -> Dict[str, Any]:
        """获取领域的Schema"""
        if domain not in self.DOMAIN_TEMPLATES:
            raise ValueError(f"未知领域: {domain}")
        return self.DOMAIN_TEMPLATES[domain]


def demo_synthesis():
    """演示数据合成"""
    synthesizer = DataSynthesizer()
    
    print("可用领域:", synthesizer.get_available_domains())
    
    # 生成企业销售领域的数据
    results = synthesizer.synthesize_from_domain("企业销售", n_samples=5)
    
    print("\n生成的样本:")
    for r in results:
        print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo_synthesis()
