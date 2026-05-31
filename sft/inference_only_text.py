import os
import json
import argparse
import torch
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from transformers import AutoProcessor, BitsAndBytesConfig
try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None
from swift.utils import seed_everything
from preprocess_data import load_tables

BASE_MODEL_ID = os.getenv("BILINGUAL_SQL_CODER_MODEL_PATH", "Qwen/Qwen3.5-4B")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Spider Text-to-SQL (Qwen3.5 text-only mode)")
    parser.add_argument("--model_type", type=str, choices=['base', 'tuned'], default='base', help="使用基础模型(base)或微调模型(tuned)")
    parser.add_argument("--base_model_id", type=str, default=BASE_MODEL_ID, help="Qwen3.5基座模型路径或Hugging Face ID")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="LoRA微调权重目录")
    parser.add_argument("--output_file", type=str, required=True, help="预测SQL的保存路径（必填）")
    parser.add_argument("--dev_file", type=str, default="spider/dev.json", help="Spider开发集数据路径")
    parser.add_argument("--tables_file", type=str, default="spider/tables.json", help="数据库表结构文件路径")
    return parser.parse_args()

def find_latest_checkpoint(checkpoint_dir):
    """(保持不变) 支持直接指定checkpoint目录 + 适配.safetensors格式LoRA权重"""
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
            raise ValueError("❌ 使用tuned模式必须通过 --checkpoint_dir 指定LoRA/DoRA权重目录")

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

def generate_predictions(args):
    seed_everything(42)

    print(f"=== 基础配置 (Qwen3.5 Text-Only Mode) ===")
    print(f"基座模型路径：{args.base_model_id}")
    print(f"模型类型：{args.model_type}")
    if args.model_type == 'tuned':
        print(f"LoRA checkpoint目录：{args.checkpoint_dir}")
    print("="*50)

    model, processor = load_qwen35_model(args.base_model_id, args.model_type, args.checkpoint_dir)
    tokenizer = get_text_tokenizer(processor)

    # 权重验证 (保持不变)
    print("\n=== 权重加载验证结果 ===")
    if isinstance(model, PeftModel):
        print(f"📌 激活的适配器名称：{model.active_adapter}")
    else:
        print("📌 当前使用纯基础模型")
    print("="*60 + "\n")

    # 准备数据
    print(f"加载数据库表结构：{args.tables_file}")
    schema_map = load_tables(args.tables_file)
    print(f"加载开发集数据：{args.dev_file}")
    with open(args.dev_file, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    predictions = []
    print("\n开始推理...")
    
    for item in tqdm(dev_data, desc="生成SQL"):
        db_id = item['db_id']
        question = item['question']
        
        if db_id not in schema_map:
            predictions.append("SELECT * FROM T") 
            continue
            
        schema_context = schema_map[db_id]
        
        # 构造 Prompt 内容
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
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for sql in predictions:
            f.write(sql + '\n')
    print(f"预测结果已保存到：{args.output_file}")

if __name__ == "__main__":
    args = parse_args()
    generate_predictions(args)
