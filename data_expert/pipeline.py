"""
双语数据处理专家 - 完整Pipeline
Full Pipeline for Bilingual Data Processing Expert
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from .api_client import QwenClient
from .data_cleaner import DataCleaner
from .translator import BilingualTranslator
from .synthesizer import DataSynthesizer
from .eval_generator import EvalDataGenerator
from .config import OUTPUT_DIR, DATA_DIR

logger = logging.getLogger(__name__)


class DataExpertPipeline:
    """数据处理专家完整Pipeline"""
    
    def __init__(self, api_key: str = None, output_dir: str = None):
        """
        初始化Pipeline
        
        Args:
            api_key: DashScope API Key
            output_dir: 输出目录
        """
        self.client = QwenClient(api_key=api_key)
        self.cleaner = DataCleaner(client=self.client)
        self.translator = BilingualTranslator(client=self.client)
        self.synthesizer = DataSynthesizer(client=self.client)
        self.eval_generator = EvalDataGenerator(client=self.client)
        
        self.output_dir = Path(output_dir or OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_spider_data(self, data_path: str) -> tuple:
        """
        加载Spider格式的数据
        
        Args:
            data_path: 数据文件路径（JSON或JSONL）
            
        Returns:
            (样本列表, Schema字典)
        """
        samples = []
        
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                else:
                    samples = data.get('data', [data])
        
        logger.info(f"加载了 {len(samples)} 个样本")
        return samples
    
    def load_schema(self, schema_path: str) -> Dict[str, Dict[str, Any]]:
        """
        加载Schema文件
        
        Args:
            schema_path: Schema文件路径
            
        Returns:
            db_id到Schema的映射字典
        """
        with open(schema_path, 'r', encoding='utf-8') as f:
            schemas = json.load(f)
        
        schema_dict = {}
        for schema in schemas:
            db_id = schema.get('db_id', schema.get('database_id', ''))
            if db_id:
                schema_dict[db_id] = schema
        
        logger.info(f"加载了 {len(schema_dict)} 个Schema")
        return schema_dict
    
    def run_cleaning_pipeline(
        self,
        samples: List[Dict[str, Any]],
        schema_dict: Dict[str, Dict[str, Any]],
        confidence_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        运行数据清洗Pipeline
        
        Args:
            samples: 样本列表
            schema_dict: Schema字典
            confidence_threshold: 置信度阈值
            
        Returns:
            清洗结果
        """
        logger.info("开始数据清洗...")
        
        # 批量清洗
        cleaned_results = self.cleaner.clean_batch(samples, schema_dict)
        
        # 过滤有效样本
        valid, invalid = self.cleaner.filter_valid_samples(
            cleaned_results, confidence_threshold
        )
        
        # 生成报告
        report = self.cleaner.generate_cleaning_report(cleaned_results)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self._save_jsonl(
            [r.get("original_sample") for r in valid if r.get("original_sample")],
            self.output_dir / f"cleaned_valid_{timestamp}.jsonl"
        )
        
        self._save_jsonl(
            [r.get("original_sample") for r in invalid if r.get("original_sample")],
            self.output_dir / f"cleaned_invalid_{timestamp}.jsonl"
        )
        
        self._save_json(report, self.output_dir / f"cleaning_report_{timestamp}.json")
        
        logger.info(f"清洗完成: 有效 {len(valid)}, 无效 {len(invalid)}")
        
        return {
            "valid": valid,
            "invalid": invalid,
            "report": report
        }
    
    def run_translation_pipeline(
        self,
        samples: List[Dict[str, Any]],
        schema_dict: Dict[str, Dict[str, Any]],
        n_variants: int = 3,
        include_dirty: bool = True
    ) -> List[Dict[str, Any]]:
        """
        运行翻译改写Pipeline
        
        Args:
            samples: 样本列表
            schema_dict: Schema字典
            n_variants: 每个样本的变体数量
            include_dirty: 是否包含脏数据变体
            
        Returns:
            翻译后的数据集
        """
        logger.info("开始翻译改写...")
        
        if include_dirty:
            dataset = self.translator.create_bilingual_dataset(
                samples, schema_dict,
                include_dirty=True,
                n_clean_variants=n_variants,
                n_dirty_variants=1
            )
        else:
            dataset = self.translator.translate_batch(
                samples, schema_dict, n_variants
            )
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_jsonl(dataset, self.output_dir / f"bilingual_dataset_{timestamp}.jsonl")
        
        logger.info(f"翻译完成: 生成 {len(dataset)} 条数据")
        
        return dataset
    
    def run_synthesis_pipeline(
        self,
        domains: List[str] = None,
        n_per_domain: int = 20,
        custom_schemas: Dict[str, Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        运行数据合成Pipeline
        
        Args:
            domains: 领域列表
            n_per_domain: 每个领域的样本数
            custom_schemas: 自定义Schema字典
            
        Returns:
            合成的数据集
        """
        logger.info("开始数据合成...")
        
        all_data = []
        
        # 使用预定义领域
        if domains:
            data = self.synthesizer.synthesize_multi_domain(n_per_domain, domains)
            all_data.extend(data)
        
        # 使用自定义Schema
        if custom_schemas:
            for db_id, schema_info in custom_schemas.items():
                schema = schema_info.get("schema", schema_info)
                domain = schema_info.get("domain", db_id)
                
                data = self.synthesizer.synthesize_balanced_dataset(
                    schema, db_id, domain, n_per_domain
                )
                all_data.extend(data)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_jsonl(all_data, self.output_dir / f"synthetic_dataset_{timestamp}.jsonl")
        
        logger.info(f"合成完成: 生成 {len(all_data)} 条数据")
        
        return all_data
    
    def run_eval_generation_pipeline(
        self,
        train_samples: List[Dict[str, Any]],
        schema_dict: Dict[str, Dict[str, Any]],
        holdout_ratio: float = 0.1
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        运行评测数据生成Pipeline
        
        Args:
            train_samples: 训练样本
            schema_dict: Schema字典
            holdout_ratio: holdout比例
            
        Returns:
            评测数据集
        """
        logger.info("开始生成评测数据...")
        
        eval_dataset = self.eval_generator.generate_eval_dataset(
            train_samples, schema_dict, holdout_ratio
        )
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 合并所有评测数据
        all_eval = []
        for key in ["paraphrase", "harder", "edge_cases"]:
            all_eval.extend(eval_dataset.get(key, []))
        
        self._save_jsonl(all_eval, self.output_dir / f"eval_dataset_{timestamp}.jsonl")
        self._save_json(
            {"holdout_indices": eval_dataset.get("holdout_indices", [])},
            self.output_dir / f"eval_metadata_{timestamp}.json"
        )
        
        logger.info(f"评测数据生成完成: 共 {len(all_eval)} 条")
        
        return eval_dataset
    
    def run_full_pipeline(
        self,
        source_data_path: str,
        schema_path: str = None,
        clean: bool = True,
        translate: bool = True,
        synthesize: bool = True,
        generate_eval: bool = True,
        synthesis_domains: List[str] = None,
        n_translation_variants: int = 3,
        n_synthesis_per_domain: int = 20
    ) -> Dict[str, Any]:
        """
        运行完整Pipeline
        
        Args:
            source_data_path: 源数据路径
            schema_path: Schema文件路径
            clean: 是否执行清洗
            translate: 是否执行翻译
            synthesize: 是否执行合成
            generate_eval: 是否生成评测数据
            synthesis_domains: 合成数据的领域列表
            n_translation_variants: 翻译变体数量
            n_synthesis_per_domain: 每领域合成数量
            
        Returns:
            完整结果
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "source": source_data_path
        }
        
        # 加载数据
        samples = self.load_spider_data(source_data_path)
        schema_dict = self.load_schema(schema_path) if schema_path else {}
        
        # 1. 数据清洗
        if clean:
            cleaning_result = self.run_cleaning_pipeline(samples, schema_dict)
            results["cleaning"] = {
                "valid_count": len(cleaning_result["valid"]),
                "invalid_count": len(cleaning_result["invalid"]),
                "report": cleaning_result["report"]
            }
            # 使用清洗后的有效数据
            samples = [r.get("original_sample") for r in cleaning_result["valid"] 
                      if r.get("original_sample")]
        
        # 2. 翻译改写
        if translate:
            translated = self.run_translation_pipeline(
                samples, schema_dict, n_translation_variants
            )
            results["translation"] = {
                "count": len(translated)
            }
        
        # 3. 数据合成
        if synthesize:
            synthesis_domains = synthesis_domains or self.synthesizer.get_available_domains()
            synthesized = self.run_synthesis_pipeline(
                domains=synthesis_domains,
                n_per_domain=n_synthesis_per_domain
            )
            results["synthesis"] = {
                "count": len(synthesized),
                "domains": synthesis_domains
            }
        
        # 4. 评测数据生成
        if generate_eval:
            eval_data = self.run_eval_generation_pipeline(samples, schema_dict)
            results["evaluation"] = {
                "paraphrase_count": len(eval_data.get("paraphrase", [])),
                "harder_count": len(eval_data.get("harder", [])),
                "edge_cases_count": len(eval_data.get("edge_cases", []))
            }
        
        # 保存总结报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_json(results, self.output_dir / f"pipeline_summary_{timestamp}.json")
        
        logger.info("完整Pipeline执行完成！")
        logger.info(f"结果已保存到: {self.output_dir}")
        
        return results
    
    def _save_json(self, data: Any, path: Path):
        """保存JSON文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _save_jsonl(self, data: List[Dict], path: Path):
        """保存JSONL文件"""
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "api_stats": self.client.get_stats(),
            "output_dir": str(self.output_dir)
        }


def create_sample_data():
    """创建示例数据用于测试"""
    samples = [
        {
            "question": "Find the name of students who have grade higher than 80",
            "sql": "SELECT name FROM students WHERE grade > 80",
            "db_id": "school"
        },
        {
            "question": "What is the average price of products in category Electronics?",
            "sql": "SELECT AVG(price) FROM products WHERE category = 'Electronics'",
            "db_id": "shop"
        },
        {
            "question": "List all orders placed in the last month",
            "sql": "SELECT * FROM orders WHERE order_date >= DATE('now', '-1 month')",
            "db_id": "ecommerce"
        }
    ]
    
    schemas = [
        {
            "db_id": "school",
            "tables": [
                {"name": "students", "columns": ["id", "name", "grade", "class_id"]}
            ]
        },
        {
            "db_id": "shop",
            "tables": [
                {"name": "products", "columns": ["id", "name", "price", "category"]}
            ]
        },
        {
            "db_id": "ecommerce",
            "tables": [
                {"name": "orders", "columns": ["id", "customer_id", "order_date", "total"]}
            ]
        }
    ]
    
    return samples, {s["db_id"]: s for s in schemas}


if __name__ == "__main__":
    # 演示Pipeline使用
    logging.basicConfig(level=logging.INFO)
    
    pipeline = DataExpertPipeline()
    
    # 创建示例数据
    samples, schema_dict = create_sample_data()
    
    print("可用领域:", pipeline.synthesizer.get_available_domains())
    print("\n运行翻译...")
    
    # 小规模演示
    translated = pipeline.translator.translate_sample(
        question_en=samples[0]["question"],
        sql=samples[0]["sql"],
        schema=schema_dict.get(samples[0]["db_id"], {}),
        db_id=samples[0]["db_id"],
        n_variants=2
    )
    
    print("翻译结果:")
    for t in translated:
        print(json.dumps(t, ensure_ascii=False, indent=2))
