"""
双语数据处理专家 - 通义千问API客户端
Qwen3 API Client for Bilingual Data Processing Expert
"""

import os
import time
import json
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("请安装openai库: pip install openai")

from .config import (
    DASHSCOPE_API_KEY, 
    DASHSCOPE_BASE_URL, 
    DEFAULT_MODEL,
    MAX_RETRIES,
    RETRY_DELAY,
    MAX_CONCURRENT_REQUESTS,
    REQUEST_DELAY,
    GENERATION_CONFIG
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QwenClient:
    """通义千问Qwen3 API客户端"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        初始化Qwen API客户端
        
        Args:
            api_key: DashScope API Key，默认从环境变量读取
            base_url: API基础URL
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or DASHSCOPE_API_KEY
        self.base_url = base_url or DASHSCOPE_BASE_URL
        
        if self.api_key == "sk-your-key-here":
            logger.warning("请设置有效的DASHSCOPE_API_KEY！可通过环境变量或直接传入。")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.request_count = 0
        self.last_request_time = 0
    
    def _rate_limit(self):
        """简单的速率限制"""
        elapsed = time.time() - self.last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self.last_request_time = time.time()
    
    def call(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        retry: int = None
    ) -> str:
        """
        调用Qwen3 API
        
        Args:
            messages: 对话消息列表
            model: 模型名称
            temperature: 温度参数（控制多样性）
            max_tokens: 最大生成token数
            top_p: nucleus sampling参数
            retry: 重试次数
            
        Returns:
            模型生成的文本
        """
        model = model or DEFAULT_MODEL
        retry = retry if retry is not None else MAX_RETRIES
        
        self._rate_limit()
        
        for attempt in range(retry + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
                self.request_count += 1
                return response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{retry + 1}): {e}")
                if attempt < retry:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise
    
    def call_with_json_output(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        **kwargs
    ) -> Any:
        """
        调用API并解析JSON输出
        
        Args:
            messages: 对话消息列表
            model: 模型名称
            **kwargs: 其他参数
            
        Returns:
            解析后的JSON对象
        """
        response = self.call(messages, model, **kwargs)
        
        # 尝试提取JSON
        try:
            # 尝试直接解析
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试从markdown代码块中提取
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                return json.loads(json_match.group(1))
            
            # 尝试找到JSON数组或对象
            json_match = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', response)
            if json_match:
                return json.loads(json_match.group(1))
            
            logger.error(f"无法解析JSON响应: {response[:500]}...")
            raise ValueError("API返回的内容无法解析为JSON")
    
    def batch_call(
        self,
        messages_list: List[List[Dict[str, str]]],
        model: str = None,
        max_workers: int = None,
        **kwargs
    ) -> List[str]:
        """
        批量并发调用API
        
        Args:
            messages_list: 多组对话消息
            model: 模型名称
            max_workers: 最大并发数
            **kwargs: 其他参数
            
        Returns:
            结果列表
        """
        max_workers = max_workers or MAX_CONCURRENT_REQUESTS
        results = [None] * len(messages_list)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.call, messages, model, **kwargs): idx
                for idx, messages in enumerate(messages_list)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"批量调用失败 (索引 {idx}): {e}")
                    results[idx] = None
        
        return results
    
    def get_task_config(self, task_type: str) -> Dict[str, Any]:
        """
        获取特定任务的配置
        
        Args:
            task_type: 任务类型 (translation/synthesis/cleaning/evaluation)
            
        Returns:
            配置字典
        """
        return GENERATION_CONFIG.get(task_type, {
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.95
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """获取API调用统计"""
        return {
            "total_requests": self.request_count
        }


def test_connection():
    """测试API连接"""
    client = QwenClient()
    try:
        response = client.call([
            {"role": "user", "content": "你好，请简短回复确认连接正常。"}
        ], max_tokens=50)
        print(f"✅ API连接成功！响应: {response}")
        return True
    except Exception as e:
        print(f"❌ API连接失败: {e}")
        return False


if __name__ == "__main__":
    test_connection()
