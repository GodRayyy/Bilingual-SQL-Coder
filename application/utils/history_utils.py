import os
import json
import glob
import pandas as pd
from datetime import datetime

# 定义历史记录存储路径
HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "history")

def get_all_conversations():
    """获取所有历史对话列表"""
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)
        
    files = glob.glob(os.path.join(HISTORY_DIR, "*.json"))
    conversations = []
    
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                conversations.append({
                    "id": data.get("id"),
                    "title": data.get("title", "未命名对话"),
                    "timestamp": data.get("timestamp", 0),
                    "time_str": data.get("time_str", ""),
                    "dataset": data.get("dataset", "Spider (English)"), # 新增字段
                    "file_path": f
                })
        except Exception:
            continue
            
    conversations.sort(key=lambda x: x["timestamp"], reverse=True)
    return conversations

def load_conversation(conversation_id):
    """加载指定 ID 的对话记录"""
    file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if "messages" in data:
                for msg in data["messages"]:
                    if "dataframe" in msg and msg["dataframe"] is not None:
                        try:
                            msg["dataframe"] = pd.DataFrame(msg["dataframe"])
                        except Exception:
                            msg["dataframe"] = None
            return data
        except Exception as e:
            print(f"Error loading conversation: {e}")
            return None
    return None

def save_conversation(conversation_id, messages, selected_dataset, selected_db):
    """
    保存对话到 JSON 文件
    新增参数: selected_dataset (记录属于哪个数据集)
    """
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)
        
    file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    
    # 生成标题
    title = "新对话"
    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"]
            title = content[:15] + "..." if len(content) > 15 else content
            break

    # 处理 DataFrame 序列化
    messages_to_save = []
    for msg in messages:
        msg_copy = msg.copy()
        if "dataframe" in msg_copy:
            df_obj = msg_copy["dataframe"]
            if df_obj is not None and isinstance(df_obj, pd.DataFrame):
                msg_copy["dataframe"] = df_obj.to_dict(orient='records')
            else:
                msg_copy["dataframe"] = None
        messages_to_save.append(msg_copy)
            
    data = {
        "id": conversation_id,
        "title": title,
        "timestamp": datetime.now().timestamp(),
        "time_str": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": selected_dataset,  # 保存数据集名称
        "selected_db": selected_db, 
        "messages": messages_to_save
    }
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Save failed: {e}")

def create_new_conversation_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def delete_conversation(conversation_id):
    file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    if os.path.exists(file_path):
        os.remove(file_path)