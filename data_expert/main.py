#!/usr/bin/env python3
"""
双语数据处理专家 - 主程序入口
Main Entry for Bilingual Data Processing Expert

使用方法:
    # 测试API连接
    python main.py test
    
    # 运行完整Pipeline
    python main.py run --data path/to/data.json --schema path/to/tables.json
    
    # 仅翻译
    python main.py translate --data path/to/data.json --n 3
    
    # 仅合成
    python main.py synthesize --domains 企业销售 学生成绩 --n 20
    
    # 生成评测数据
    python main.py eval --data path/to/train.json --schema path/to/tables.json
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_expert import (
    QwenClient,
    QuotaExhaustedError,
    APIServiceError,
    DataCleaner,
    BilingualTranslator,
    DataSynthesizer,
    EvalDataGenerator,
    DataExpertPipeline
)
from data_expert.api_client import test_connection

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_test(args):
    """测试API连接"""
    print("=" * 50)
    print("测试通义千问API连接...")
    print("=" * 50)
    
    success = test_connection()
    
    if success:
        print("\n✅ API连接测试成功！可以开始使用数据处理专家。")
    else:
        print("\n❌ API连接失败，请检查：")
        print("1. 是否设置了环境变量 DASHSCOPE_API_KEY")
        print("2. API Key是否有效")
        print("3. 网络连接是否正常")
        print("\n设置方法: export DASHSCOPE_API_KEY='sk-your-key-here'")
    
    return success


def cmd_translate(args):
    """翻译命令"""
    print("=" * 50)
    print("开始双语翻译...")
    print("=" * 50)
    
    pipeline = DataExpertPipeline(output_dir=args.output)
    
    # 加载数据
    samples = pipeline.load_spider_data(args.data)
    schema_dict = pipeline.load_schema(args.schema) if args.schema else {}
    
    # 限制数量
    if args.limit:
        samples = samples[:args.limit]
    
    # 翻译
    result = pipeline.run_translation_pipeline(
        samples, schema_dict,
        n_variants=args.n,
        include_dirty=args.dirty
    )
    
    print(f"\n✅ 翻译完成！生成 {len(result)} 条数据")
    print(f"结果保存在: {pipeline.output_dir}")
    
    return result


def cmd_synthesize(args):
    """合成命令"""
    print("=" * 50)
    print("开始数据合成...")
    print("=" * 50)
    
    pipeline = DataExpertPipeline(output_dir=args.output)
    
    # 显示可用领域
    available = pipeline.synthesizer.get_available_domains()
    print(f"可用领域: {available}")
    
    # 确定要使用的领域
    domains = args.domains if args.domains else available
    
    # 合成
    result = pipeline.run_synthesis_pipeline(
        domains=domains,
        n_per_domain=args.n
    )
    
    print(f"\n✅ 合成完成！生成 {len(result)} 条数据")
    print(f"结果保存在: {pipeline.output_dir}")
    
    return result


def cmd_clean(args):
    """清洗命令"""
    print("=" * 50)
    print("开始数据清洗...")
    print("=" * 50)
    
    pipeline = DataExpertPipeline(output_dir=args.output)
    
    # 加载数据
    samples = pipeline.load_spider_data(args.data)
    schema_dict = pipeline.load_schema(args.schema) if args.schema else {}
    
    # 限制数量
    if args.limit:
        samples = samples[:args.limit]
    
    # 清洗
    result = pipeline.run_cleaning_pipeline(
        samples, schema_dict,
        confidence_threshold=args.threshold
    )
    
    print(f"\n✅ 清洗完成！")
    print(f"  有效样本: {len(result['valid'])}")
    print(f"  无效样本: {len(result['invalid'])}")
    print(f"结果保存在: {pipeline.output_dir}")
    
    return result


def cmd_eval(args):
    """生成评测数据命令"""
    print("=" * 50)
    print("开始生成评测数据...")
    print("=" * 50)
    
    pipeline = DataExpertPipeline(output_dir=args.output)
    
    # 加载数据
    samples = pipeline.load_spider_data(args.data)
    schema_dict = pipeline.load_schema(args.schema) if args.schema else {}
    
    # 限制数量
    if args.limit:
        samples = samples[:args.limit]
    
    # 生成评测数据
    result = pipeline.run_eval_generation_pipeline(
        samples, schema_dict,
        holdout_ratio=args.ratio
    )
    
    total = sum(len(v) for k, v in result.items() if isinstance(v, list))
    print(f"\n✅ 评测数据生成完成！共 {total} 条")
    print(f"结果保存在: {pipeline.output_dir}")
    
    return result


def cmd_run(args):
    """运行完整Pipeline"""
    print("=" * 50)
    print("运行完整数据处理Pipeline...")
    print("=" * 50)
    
    pipeline = DataExpertPipeline(output_dir=args.output)
    
    try:
        result = pipeline.run_full_pipeline(
            source_data_path=args.data,
            schema_path=args.schema,
            clean=not args.no_clean,
            translate=not args.no_translate,
            synthesize=not args.no_synthesize,
            generate_eval=not args.no_eval,
            synthesis_domains=args.domains,
            n_translation_variants=args.n_translate,
            n_synthesis_per_domain=args.n_synthesize
        )
        
        if result.get("status") == "completed":
            print("\n✅ 完整Pipeline执行完成！")
        elif result.get("status") == "quota_exhausted":
            print("\n⚠️ API配额已耗尽，Pipeline已中断")
            print("已处理的数据已保存，请查看 checkpoints 目录")
            print("\n恢复执行命令:")
            print(f"  python main.py resume --checkpoint <checkpoint_file>")
        else:
            print(f"\n❌ Pipeline执行出错: {result.get('status')}")
            
        print(f"结果保存在: {pipeline.output_dir}")
        print("\n执行摘要:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        return result
        
    except QuotaExhaustedError as e:
        print(f"\n❌ API配额耗尽: {e}")
        print("已处理的数据已自动保存到 checkpoints 目录")
        print("\n请查看错误报告获取恢复命令:")
        print(f"  ls {pipeline.checkpoint_dir}/error_report_*.json")
        return None


def cmd_resume(args):
    """从断点恢复执行"""
    print("=" * 50)
    print("从断点恢复执行...")
    print("=" * 50)
    
    pipeline = DataExpertPipeline(output_dir=args.output)
    
    try:
        result = pipeline.resume_from_checkpoint(args.checkpoint)
        
        if result.get("status") == "completed":
            print(f"\n✅ 恢复执行完成！")
            print(f"  - 总计处理: {result.get('total_count')} 条")
            print(f"  - 从断点恢复: {result.get('resumed_from_count')} 条已存在")
        else:
            print(f"\n⚠️ 恢复执行状态: {result.get('status')}")
        
        print(f"结果保存在: {pipeline.output_dir}")
        return result
        
    except Exception as e:
        print(f"\n❌ 恢复执行失败: {e}")
        return None


def cmd_list_checkpoints(args):
    """列出所有检查点"""
    print("=" * 50)
    print("可用的检查点:")
    print("=" * 50)
    
    from data_expert.config import CHECKPOINT_DIR
    checkpoint_dir = Path(CHECKPOINT_DIR)
    
    if not checkpoint_dir.exists():
        print("暂无检查点")
        return
    
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.json"), reverse=True)
    
    if not checkpoints:
        print("暂无检查点")
        return
    
    for cp in checkpoints:
        with open(cp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"\n📁 {cp.name}")
        print(f"   时间: {data.get('timestamp')}")
        print(f"   阶段: {data.get('stage')}")
        print(f"   已处理: {data.get('processed_count')} 条")
        print(f"   剩余: {data.get('remaining_count')} 条")
    
    # 列出错误报告
    errors = sorted(checkpoint_dir.glob("error_report_*.json"), reverse=True)
    if errors:
        print("\n" + "=" * 50)
        print("错误报告:")
        print("=" * 50)
        for err in errors[:5]:  # 只显示最近5个
            with open(err, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"\n⚠️ {err.name}")
            print(f"   时间: {data.get('timestamp')}")
            print(f"   阶段: {data.get('stage')}")
            print(f"   错误类型: {data.get('error_type')}")
            print(f"   进度: {data.get('progress_percentage')}")


def cmd_demo(args):
    """运行演示"""
    print("=" * 50)
    print("运行数据处理专家演示...")
    print("=" * 50)
    
    # 测试连接
    if not test_connection():
        return
    
    client = QwenClient()
    
    # 演示翻译
    print("\n--- 1. 翻译演示 ---")
    translator = BilingualTranslator(client)
    
    sample = {
        "question": "Find the names of students who scored above 90",
        "sql": "SELECT name FROM students WHERE score > 90",
        "db_id": "school"
    }
    
    schema = {
        "tables": [{"name": "students", "columns": ["id", "name", "score"]}]
    }
    
    translated = translator.translate_sample(
        question_en=sample["question"],
        sql=sample["sql"],
        schema=schema,
        db_id=sample["db_id"],
        n_variants=2
    )
    
    print("翻译结果:")
    for t in translated:
        print(f"  中文: {t.get('question_zh', 'N/A')}")
        print(f"  SQL: {t.get('sql', 'N/A')}")
        print()
    
    # 演示合成
    print("\n--- 2. 数据合成演示 ---")
    synthesizer = DataSynthesizer(client)
    print(f"可用领域: {synthesizer.get_available_domains()}")
    
    synthesized = synthesizer.synthesize_from_domain("学生成绩", n_samples=3)
    print(f"生成了 {len(synthesized)} 条合成数据")
    for s in synthesized[:2]:
        print(f"  问题: {s.get('question_zh', 'N/A')}")
        print(f"  SQL: {s.get('sql', 'N/A')}")
        print()
    
    print("\n✅ 演示完成！")
    print("\n使用完整Pipeline的示例命令:")
    print("  python main.py run --data your_data.json --schema tables.json")


def main():
    parser = argparse.ArgumentParser(
        description="双语数据处理专家 - Text-to-SQL数据处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 测试API连接
  python main.py test
  
  # 运行演示
  python main.py demo
  
  # 翻译数据
  python main.py translate --data spider_train.json --n 3
  
  # 合成数据
  python main.py synthesize --domains 企业销售 学生成绩 --n 20
  
  # 运行完整Pipeline
  python main.py run --data spider_train.json --schema tables.json
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # test命令
    test_parser = subparsers.add_parser("test", help="测试API连接")
    
    # demo命令
    demo_parser = subparsers.add_parser("demo", help="运行演示")
    
    # translate命令
    translate_parser = subparsers.add_parser("translate", help="翻译数据")
    translate_parser.add_argument("--data", required=True, help="输入数据文件路径")
    translate_parser.add_argument("--schema", help="Schema文件路径")
    translate_parser.add_argument("--output", default="./generated_data", help="输出目录")
    translate_parser.add_argument("--n", type=int, default=3, help="每个样本的变体数量")
    translate_parser.add_argument("--limit", type=int, help="处理的样本数量限制")
    translate_parser.add_argument("--dirty", action="store_true", help="包含脏数据变体")
    
    # synthesize命令
    synth_parser = subparsers.add_parser("synthesize", help="合成数据")
    synth_parser.add_argument("--domains", nargs="+", help="领域列表")
    synth_parser.add_argument("--output", default="./generated_data", help="输出目录")
    synth_parser.add_argument("--n", type=int, default=20, help="每个领域的样本数量")
    
    # clean命令
    clean_parser = subparsers.add_parser("clean", help="清洗数据")
    clean_parser.add_argument("--data", required=True, help="输入数据文件路径")
    clean_parser.add_argument("--schema", help="Schema文件路径")
    clean_parser.add_argument("--output", default="./generated_data", help="输出目录")
    clean_parser.add_argument("--threshold", type=float, default=0.8, help="置信度阈值")
    clean_parser.add_argument("--limit", type=int, help="处理的样本数量限制")
    
    # eval命令
    eval_parser = subparsers.add_parser("eval", help="生成评测数据")
    eval_parser.add_argument("--data", required=True, help="训练数据文件路径")
    eval_parser.add_argument("--schema", help="Schema文件路径")
    eval_parser.add_argument("--output", default="./generated_data", help="输出目录")
    eval_parser.add_argument("--ratio", type=float, default=0.1, help="holdout比例")
    eval_parser.add_argument("--limit", type=int, help="处理的样本数量限制")
    
    # run命令
    run_parser = subparsers.add_parser("run", help="运行完整Pipeline")
    run_parser.add_argument("--data", required=True, help="输入数据文件路径")
    run_parser.add_argument("--schema", help="Schema文件路径")
    run_parser.add_argument("--output", default="./generated_data", help="输出目录")
    run_parser.add_argument("--domains", nargs="+", help="合成数据的领域列表")
    run_parser.add_argument("--n-translate", type=int, default=3, help="翻译变体数量")
    run_parser.add_argument("--n-synthesize", type=int, default=20, help="每领域合成数量")
    run_parser.add_argument("--no-clean", action="store_true", help="跳过清洗步骤")
    run_parser.add_argument("--no-translate", action="store_true", help="跳过翻译步骤")
    run_parser.add_argument("--no-synthesize", action="store_true", help="跳过合成步骤")
    run_parser.add_argument("--no-eval", action="store_true", help="跳过评测数据生成")
    
    # resume命令 - 从断点恢复
    resume_parser = subparsers.add_parser("resume", help="从断点恢复执行")
    resume_parser.add_argument("--checkpoint", required=True, help="检查点文件路径")
    resume_parser.add_argument("--output", default="./generated_data", help="输出目录")
    
    # checkpoints命令 - 列出检查点
    checkpoints_parser = subparsers.add_parser("checkpoints", help="列出所有检查点和错误报告")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # 执行对应命令
    commands = {
        "test": cmd_test,
        "demo": cmd_demo,
        "translate": cmd_translate,
        "synthesize": cmd_synthesize,
        "clean": cmd_clean,
        "eval": cmd_eval,
        "run": cmd_run,
        "resume": cmd_resume,
        "checkpoints": cmd_list_checkpoints
    }
    
    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
