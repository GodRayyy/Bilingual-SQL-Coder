进入系统文件夹
cd application

默认模型路径
export BILINGUAL_SQL_CODER_MODEL_PATH=/path/to/Qwen3.5-4B

如需加载重新基于 Qwen3.5 微调的 LoRA/DoRA 适配器
export BILINGUAL_SQL_CODER_ADAPTER_PATH=/path/to/qwen35/checkpoint

安装环境
pip install -r requirements.txt

运行系统命令：
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.fileWatcherType none
