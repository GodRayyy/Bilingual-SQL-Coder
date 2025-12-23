import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import streamlit as st
from config import settings

@st.cache_resource
def load_model_and_tokenizer():
    """加载模型和分词器 (只运行一次)"""
    print("正在加载基座模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        settings.BASE_MODEL_PATH, 
        trust_remote_code=True
    )
    
    # 加载基座模型 (推荐使用 4-bit 或 8-bit 量化以节省显存，根据你的环境调整)
    # 如果显存足够，可以去掉 load_in_4bit 等参数
    model = AutoModelForCausalLM.from_pretrained(
        settings.BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print(f"正在加载 DoRA 适配器: {settings.ADAPTER_PATH}")
    # 加载微调后的权重
    if settings.ADAPTER_PATH and os.path.exists(settings.ADAPTER_PATH):
        model = PeftModel.from_pretrained(model, settings.ADAPTER_PATH)
        
    model.eval()
    return model, tokenizer

def generate_sql_query(model, tokenizer, question, schema):
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
    
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=settings.MAX_NEW_TOKENS,
            temperature=settings.TEMPERATURE,
            do_sample=False  # Text2SQL 建议使用贪婪搜索或极低温度
        )
        
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 清洗输出 (去除可能的 Markdown 标记)
    clean_sql = response.replace("```sql", "").replace("```", "").strip()
    return clean_sql