import os
import json
import argparse
import torch
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from swift.utils import seed_everything
from preprocess_data import load_tables

BASE_MODEL_ID = "/data0/dywang/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507" 

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Spider Text-to-SQL (纯文本模型版)")
    parser.add_argument("--model_type", type=str, choices=['base', 'tuned'], default='base', help="使用基础模型(base)或微调模型(tuned)")
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

def generate_predictions(args):
    seed_everything(42)
    
    print(f"=== 基础配置 (Text-Only Mode) ===")
    print(f"基座模型路径：{BASE_MODEL_ID}")
    print(f"模型类型：{args.model_type}")
    if args.model_type == 'tuned':
        print(f"LoRA checkpoint目录：{args.checkpoint_dir}")
    print("="*50)
    
    # 模型加载配置
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

    # [修改点 3] 加载模型逻辑：使用 AutoModelForCausalLM
    print("\n=== 加载模型 ===")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    
    if args.model_type == 'base':
        # 纯文本模型使用 AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **model_kwargs)
        print("✅ 基础模型加载完成")
    else:
        if not args.checkpoint_dir:
            raise ValueError("❌ 使用tuned模式必须通过 --checkpoint_dir 指定LoRA权重目录")
            
        ckpt_path = find_latest_checkpoint(args.checkpoint_dir)
        print(f"LoRA权重路径：{ckpt_path}")
        
        peft_config = PeftConfig.from_pretrained(ckpt_path)
        # 加载基座
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **model_kwargs)
        # 加载LoRA
        model = PeftModel.from_pretrained(
            model,
            ckpt_path,
            device_map='auto',
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print("✅ 微调模型（基座+LoRA）加载完成")

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
        
        # [修改点 4] 使用 tokenizer.apply_chat_template 替代手动字符串拼接
        # 这是纯文本大模型（Qwen2/Qwen2.5/Qwen3）的标准用法
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        # apply_chat_template 会自动添加 <|im_start|>, <|im_end|> 等特殊 token
        text_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True # 这一步会自动添加 <|assistant|> 或等效的引导符
        )
        
        inputs = tokenizer(
            text_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096 # 纯文本模型通常可以支持更长的上下文，根据显存调整
        )
        
        input_ids = inputs['input_ids'].to(model.device)
        attention_mask = inputs['attention_mask'].to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.01, # 推理时对于代码生成，建议使用极低的 temperature 以保证确定性
                do_sample=False,  # 代码生成通常不需要采样，或者采样范围很小
            )
        
        # 解码
        input_len = input_ids.shape[1]
        output_ids = generated_ids[0][input_len:]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # 后处理（保持不变）
        cleaned_sql = response.strip()
        if "```sql" in cleaned_sql:
            cleaned_sql = cleaned_sql.split("```sql")[1].split("```")[0].strip()
        elif "```" in cleaned_sql:
            cleaned_sql = cleaned_sql.split("```")[0].strip()
        cleaned_sql = cleaned_sql.replace('\n', ' ')
        
        predictions.append(cleaned_sql)
    
    print(f"\n推理完成！共生成 {len(predictions)} 条SQL语句")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for sql in predictions:
            f.write(sql + '\n')
    print(f"预测结果已保存到：{args.output_file}")

if __name__ == "__main__":
    args = parse_args()
    generate_predictions(args)