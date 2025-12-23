# Bilingual-SQL-Coder

A comprehensive Text-to-SQL framework supporting bilingual (English & Chinese) datasets with training, data processing, deployment, and evaluation capabilities.

---

### 😮新发布！
- 权重已经开源到👐Hugging Face：https://huggingface.co/GodRayyyy/Bilingual-SQL-Coder

---

[English](#english) | [中文](#中文)

## English

### Overview

Bilingual-SQL-Coder is an end-to-end solution for Text-to-SQL tasks, supporting:
- 🧹 **Data Processing**: Bilingual translation, data cleaning, and synthesis
- 🎓 **Model Training**: SFT (Supervised Fine-Tuning) with LoRA/DoRA on large language models
- 🚀 **Application**: Web-based SQL generation interface
- 📊 **Evaluation**: Comprehensive evaluation metrics and data generation

### Project Structure

```
Bilingual-SQL-Coder/
├── data_expert/           # Bilingual data processing expert
│   ├── main.py
│   ├── pipeline.py
│   ├── translator.py      # Bilingual translation
│   ├── synthesizer.py     # Data synthesis
│   ├── data_cleaner.py    # Data cleaning
│   └── eval_generator.py  # Evaluation data generation
├── sft/                   # Supervised Fine-Tuning
│   ├── train_only_text_DoRA_all.sh
│   ├── inference_only_text.py
│   └── data/
├── application/           # Web UI & API
│   ├── app.py            # Streamlit web interface
│   └── config/
├── evaluation/           # Evaluation tools
│   ├── run_full_evaluation.py
│   ├── universal_evaluation.py
│   └── structured_sql_converter.py
└── requirements.txt      # Project dependencies
```

### Quick Start

#### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/GodRayyy/Bilingual-SQL-Coder.git
cd Bilingual-SQL-Coder

# Install dependencies
pip install -r requirements.txt
```

#### 2. Data Processing (data_expert)

The data_expert module provides comprehensive data processing:

```bash
cd data_expert

# Set API Key for Qwen API
export DASHSCOPE_API_KEY="sk-your-key-here"

# Translate data to bilingual format
python main.py translate --data spider_train.json --schema tables.json

# Synthesize data
python main.py synthesize --domains "enterprise sales" "student grades" --n 20

# Clean data
python main.py clean --data dirty_data.json

# Generate evaluation data
python main.py eval --data train.json --schema tables.json

# Run complete pipeline
python main.py run --data spider_train.json --schema tables.json
```

**Features:**
- 🌐 Bilingual translation (English ↔ Chinese)
- 🔧 Data synthesis based on Schema
- 🧹 SQL syntax error detection and fixing
- 📊 Evaluation data generation
- 🎯 Real-world data augmentation (typos, colloquial language)

#### 3. Model Training (sft)

Train models using LoRA/DoRA on Qwen models:

```bash
cd sft

# Modify training configuration in train_only_text_DoRA_all.sh
# Set your model path, data path, and training parameters

# Run training
bash train_only_text_DoRA_all.sh
```

**Configuration options in the shell script:**
- `CUDA_VISIBLE_DEVICES`: GPU selection
- `MODEL_ID`: Base model path
- `TRAIN_DATA`: Training data path
- `EVAL_DATA`: Evaluation data path

#### 4. Inference

Generate SQL from natural language questions:

```bash
cd sft

# Inference with trained model
python inference_only_text.py \
  --model_type tuned \
  --checkpoint_dir ./path/to/checkpoint \
  --output_file predictions.json
```

#### 5. Web Application (application)

Deploy the interactive web interface:

```bash
cd application

# Run Streamlit app
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Access the interface at: http://localhost:8501

#### 6. Evaluation (evaluation)

Evaluate model performance:

```bash
cd evaluation

# Run full evaluation
python run_full_evaluation.py \
  --prediction_file predictions.json \
  --gold_file dev_gold.sql
```

### Supported Models

- **Qwen3-4B-Instruct**: Lightweight, fast inference
- **Qwen3-8B-Instruct**: Balanced performance and speed
- **Qwen3-VL-8B-Instruct**: Vision-Language support (for future features)

### Dataset Support

- **Spider**: English database-agnostic text-to-SQL benchmark
- **CSpider**: Chinese version of Spider
- **Custom Datasets**: Support for custom database schemas

### API Key Configuration

For data processing features requiring LLM API:

1. Visit [Alibaba Cloud DashScope](https://dashscope.aliyuncs.com/)
2. Create an account and enable DashScope service
3. Generate API Key
4. Set environment variable: `export DASHSCOPE_API_KEY="sk-your-key"`

### Citation

If you use Bilingual-SQL-Coder in your research, please cite:

```bibtex
@software{bilingual_sql_coder,
  title={Bilingual-SQL-Coder: A Text-to-SQL Framework},
  author={Tianyu Gao, Deyang Wang, Yiwen Ma},
  year={2025},
  url={https://github.com/GodRayyy/Bilingual-SQL-Coder}
}
```

### License

MIT License - see LICENSE file for details

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 中文

### 项目介绍

Bilingual-SQL-Coder 是一个端到端的Text-to-SQL任务解决方案，支持：
- 🧹 **数据处理**：双语翻译、数据清洗和合成
- 🎓 **模型训练**：在大型语言模型上进行SFT（有监督微调）
- 🚀 **应用部署**：基于Web的SQL生成界面
- 📊 **效果评估**：完整的评估指标和数据生成

### 项目结构

```
Bilingual-SQL-Coder/
├── data_expert/           # 双语数据处理专家
│   ├── main.py
│   ├── pipeline.py
│   ├── translator.py      # 双语翻译
│   ├── synthesizer.py     # 数据合成
│   ├── data_cleaner.py    # 数据清洗
│   └── eval_generator.py  # 评测数据生成
├── sft/                   # 有监督微调
│   ├── train_only_text_DoRA_all.sh
│   ├── inference_only_text.py
│   └── data/
├── application/           # Web界面 & API
│   ├── app.py            # Streamlit网页界面
│   └── config/
├── evaluation/           # 评估工具
│   ├── run_full_evaluation.py
│   ├── universal_evaluation.py
│   └── structured_sql_converter.py
└── requirements.txt      # 项目依赖
```

### 快速开始

#### 1. 环境配置

```bash
# 克隆仓库
git clone https://github.com/GodRayyy/Bilingual-SQL-Coder.git
cd Bilingual-SQL-Coder

# 安装依赖
pip install -r requirements.txt
```

#### 2. 数据处理 (data_expert)

使用data_expert模块进行数据处理：

```bash
cd data_expert

# 设置通义千问API Key
export DASHSCOPE_API_KEY="sk-your-key-here"

# 翻译数据为双语格式
python main.py translate --data spider_train.json --schema tables.json

# 合成数据
python main.py synthesize --domains "企业销售" "学生成绩" --n 20

# 清洗数据
python main.py clean --data dirty_data.json

# 生成评测数据
python main.py eval --data train.json --schema tables.json

# 运行完整Pipeline
python main.py run --data spider_train.json --schema tables.json
```

**主要功能：**
- 🌐 双语翻译（英文 ↔ 中文）
- 🔧 基于Schema的数据合成
- 🧹 SQL语法错误检测和修复
- 📊 评测数据生成
- 🎯 真实场景数据增强（错别字、口语化等）

#### 3. 模型训练 (sft)

使用LoRA/DoRA在Qwen模型上进行微调：

```bash
cd sft

# 修改train_only_text_DoRA_all.sh中的配置
# 设置模型路径、数据路径和训练参数

# 开始训练
bash train_only_text_DoRA_all.sh
```

**shell脚本中的配置选项：**
- `CUDA_VISIBLE_DEVICES`: GPU设备选择
- `MODEL_ID`: 基础模型路径
- `TRAIN_DATA`: 训练数据路径
- `EVAL_DATA`: 评估数据路径

#### 4. 模型推理

从自然语言问题生成SQL：

```bash
cd sft

# 使用微调后的模型进行推理
python inference_only_text.py \
  --model_type tuned \
  --checkpoint_dir ./path/to/checkpoint \
  --output_file predictions.json
```

#### 5. Web应用 (application)

部署交互式Web界面：

```bash
cd application

# 运行Streamlit应用
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

访问地址: http://localhost:8501

#### 6. 效果评估 (evaluation)

评估模型性能：

```bash
cd evaluation

# 运行完整评估
python run_full_evaluation.py \
  --prediction_file predictions.json \
  --gold_file dev_gold.sql
```

### 支持的模型

- **Qwen3-4B-Instruct**: 轻量级，推理速度快
- **Qwen3-8B-Instruct**: 性能与速度平衡
- **Qwen3-VL-8B-Instruct**: 视觉-语言支持（未来功能）

### 支持的数据集

- **Spider**: 英文数据库无关Text-to-SQL基准
- **CSpider**: Spider的中文版本
- **自定义数据集**: 支持自定义数据库Schema

### API Key 配置

对于需要LLM API的数据处理功能：

1. 访问 [阿里云DashScope](https://dashscope.aliyuncs.com/)
2. 创建账户并开通DashScope服务
3. 生成API Key
4. 设置环境变量: `export DASHSCOPE_API_KEY="sk-your-key"`

### 引用

如果您在研究中使用了Bilingual-SQL-Coder，请引用：

```bibtex
@software{bilingual_sql_coder,
  title={Bilingual-SQL-Coder: A Text-to-SQL Framework},
  author={Tianyu Gao, Deyang Wang, Yiwen Ma},
  year={2025},
  url={https://github.com/GodRayyy/Bilingual-SQL-Coder}
}
```

### License

MIT License - 详见LICENSE文件

### 贡献

欢迎提交Pull Request！

