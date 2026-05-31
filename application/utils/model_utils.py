import os
import torch
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # transformers<4.57 does not include Qwen3.5 auto classes.
    AutoModelForImageTextToText = None
from peft import PeftModel
import streamlit as st
from config import settings

@st.cache_resource
def load_model_and_tokenizer():
    """加载 Qwen3.5 模型和 processor (只运行一次)"""
    if AutoModelForImageTextToText is None:
        raise ImportError(
            "当前 transformers 版本不支持 Qwen3.5。"
            "请升级到支持 AutoModelForImageTextToText 的版本（建议 >=4.57）。"
        )

    print("正在加载 Qwen3.5 processor...")
    processor = AutoProcessor.from_pretrained(
        settings.BASE_MODEL_PATH,
        trust_remote_code=True
    )

    print("正在加载 Qwen3.5 基座模型...")
    model = AutoModelForImageTextToText.from_pretrained(
        settings.BASE_MODEL_PATH,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True
    )

    if settings.ADAPTER_PATH and os.path.exists(settings.ADAPTER_PATH):
        print(f"正在加载 Qwen3.5 DoRA/LoRA 适配器: {settings.ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, settings.ADAPTER_PATH)
    elif settings.ADAPTER_PATH:
        print(f"适配器路径不存在，跳过加载: {settings.ADAPTER_PATH}")

    model.eval()
    return model, processor


def _prepare_qwen35_inputs(processor, messages, device):
    """使用 Qwen3.5 chat template 构造纯文本推理输入。"""
    try:
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False
        ).to(device)
    except TypeError:
        # 兼容旧版 processor：无法传入 enable_thinking 时仍保留标准 chat template。
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device)


def _clean_sql_output(text):
    """清洗 Qwen3.5 输出中的思考块和 Markdown 代码块。"""
    clean_sql = text.strip()
    if "</think>" in clean_sql:
        clean_sql = clean_sql.split("</think>", 1)[1].strip()
    clean_sql = clean_sql.replace("<think>", "").replace("</think>", "").strip()
    if "```sql" in clean_sql:
        clean_sql = clean_sql.split("```sql", 1)[1].split("```", 1)[0].strip()
    elif "```" in clean_sql:
        clean_sql = clean_sql.split("```", 1)[1].split("```", 1)[0].strip()
    return clean_sql.strip()


def generate_sql_query(model, processor, question, schema):
    """
    构造 Prompt 并生成 SQL
    """
    # 构造符合训练时格式的 Prompt
    # 注意：这里需要根据你训练时的 template 进行微调
    system_prompt = "你是一个专业的数据库专家。请根据给定的数据库 Schema，将用户的问题转换为可执行的 SQL 查询语句。"
    
    user_content = f"""数据库 Schema:
{schema}

问题: {question}
SQL:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    model_inputs = _prepare_qwen35_inputs(processor, messages, model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=settings.MAX_NEW_TOKENS,
            temperature=settings.TEMPERATURE,
            do_sample=False,
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return _clean_sql_output(response)
