import os
import sys
import time
import streamlit as st
import pandas as pd

# ---------------------------------------------------------
# 1. 基础环境配置
# ---------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import settings
# 设置显卡
os.environ["CUDA_VISIBLE_DEVICES"] = settings.CUDA_DEVICE

# ---------------------------------------------------------
# 2. 导入工具包
# ---------------------------------------------------------
from utils import db_utils, model_utils, history_utils

# ---------------------------------------------------------
# 3. 页面初始化与自定义样式
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bilingual-SQL-Coder",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入自定义 CSS 以美化界面
st.markdown("""
<style>
    /* 隐藏 Streamlit 默认菜单 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 聊天气泡样式微调 */
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    
    /* 侧边栏标题样式 */
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 10px;
    }
    
    /* 数据库状态标签 */
    .db-status {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. Session State 初始化
# ---------------------------------------------------------
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = history_utils.create_new_conversation_id()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_dataset" not in st.session_state:
    st.session_state.selected_dataset = settings.DEFAULT_DATASET
if "selected_db_index" not in st.session_state:
    st.session_state.selected_db_index = 0

# ---------------------------------------------------------
# 5. 核心逻辑函数
# ---------------------------------------------------------

def is_sql_statement(text):
    """判断生成的文本是否像 SQL 语句"""
    if not text: return False
    clean_text = text.strip().lower()
    sql_keywords = ["select", "with", "show", "pragma", "describe", "explain", "create", "insert"]
    return any(clean_text.startswith(kw) for kw in sql_keywords)

def switch_chat(chat_id):
    """切换对话并恢复环境"""
    st.session_state.current_chat_id = chat_id
    saved_data = history_utils.load_conversation(chat_id)
    if saved_data:
        st.session_state.messages = saved_data.get("messages", [])
        saved_dataset = saved_data.get("dataset")
        if saved_dataset and saved_dataset in settings.DATASET_CONFIG:
            st.session_state.selected_dataset = saved_dataset
            st.session_state.selected_db_index = 0
    else:
        st.session_state.messages = []

def create_new_chat():
    new_id = history_utils.create_new_conversation_id()
    st.session_state.current_chat_id = new_id
    st.session_state.messages = []

def delete_chat(chat_id):
    history_utils.delete_conversation(chat_id)
    if st.session_state.current_chat_id == chat_id:
        create_new_chat()
    st.rerun()

# ---------------------------------------------------------
# 6. 侧边栏布局 (UI 重构)
# ---------------------------------------------------------
with st.sidebar:
    # 顶部 Logo 区域
    st.title("🕸️ Bilingual-SQL-Coder")
    st.markdown("Based on **Qwen3-4B-DoRA**")
    
    # 状态指示器
    with st.spinner("正在唤醒模型..."):
        try:
            model, tokenizer = model_utils.load_model_and_tokenizer()
            st.success(f"🟢 系统在线 (GPU {settings.CUDA_DEVICE})")
        except Exception as e:
            st.error(f"🔴 模型离线: {e}")
            st.stop()
            
    st.divider()

    # --- 区域 1: 数据源配置 ---
    st.markdown('<div class="sidebar-title">⚙️ 数据源配置</div>', unsafe_allow_html=True)
    
    # 数据集选择
    dataset_names = list(settings.DATASET_CONFIG.keys())
    try:
        curr_idx = dataset_names.index(st.session_state.selected_dataset)
    except ValueError:
        curr_idx = 0

    selected_dataset_name = st.selectbox(
        "📚 数据集 (Dataset)",
        dataset_names,
        index=curr_idx,
        key="dataset_select"
    )

    # 状态同步
    if selected_dataset_name != st.session_state.selected_dataset:
        st.session_state.selected_dataset = selected_dataset_name
        st.session_state.selected_db_index = 0
        st.rerun()

    # 获取当前配置
    curr_conf = settings.DATASET_CONFIG[selected_dataset_name]
    
    # 数据库选择
    db_list = db_utils.get_all_databases(curr_conf["path"], mode=curr_conf["mode"])
    
    if not db_list:
        st.error(f"路径为空: {curr_conf['path']}")
        selected_db = None
    else:
        selected_db = st.selectbox(
            "🗄️ 目标数据库 (Database)", 
            db_list, 
            index=st.session_state.selected_db_index
        )

    # 数据预览折叠面板
    if selected_db:
        with st.expander("👀 快速查看表数据"):
            try:
                conn = db_utils.get_db_connection(curr_conf["path"], selected_db, mode=curr_conf["mode"])
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                table_view = st.selectbox("选择表名:", tables)
                if table_view:
                    # 仅显示前3行
                    df_preview = pd.read_sql_query(f"SELECT * FROM `{table_view}` LIMIT 3", conn)
                    st.dataframe(df_preview, hide_index=True, use_container_width=True)
                    cnt = cursor.execute(f"SELECT count(*) FROM `{table_view}`").fetchone()[0]
                    st.caption(f"总行数: {cnt}")
                conn.close()
            except Exception:
                st.warning("无法预览数据")

    st.divider()

    # --- 区域 2: 会话管理 ---
    st.markdown('<div class="sidebar-title">💬 历史会话</div>', unsafe_allow_html=True)
    
    if st.button("➕ 开始新对话", type="primary", use_container_width=True):
        create_new_chat()
        st.rerun()

    history_list = history_utils.get_all_conversations()
    
    # 使用 container 固定高度 (可选，streamli 会自动处理滚动)
    with st.container():
        if not history_list:
            st.caption("暂无记录")
        else:
            for chat in history_list:
                c1, c2 = st.columns([0.8, 0.2])
                
                # 标题处理
                ds_short = chat['dataset'].split(" ")[0]
                is_active = (chat['id'] == st.session_state.current_chat_id)
                prefix = "📂" if is_active else "📄"
                # 截断标题
                safe_title = (chat['title'][:14] + '..') if len(chat['title']) > 14 else chat['title']
                
                with c1:
                    if st.button(
                        f"{prefix} [{ds_short}] {safe_title}", 
                        key=f"nav_{chat['id']}", 
                        use_container_width=True,
                        help=f"时间: {chat['time_str']}\n完整标题: {chat['title']}"
                    ):
                        switch_chat(chat['id'])
                        st.rerun()
                with c2:
                    if st.button("🗑️", key=f"del_{chat['id']}", help="删除"):
                        delete_chat(chat['id'])

# ---------------------------------------------------------
# 7. 主界面逻辑
# ---------------------------------------------------------

# 顶部标题栏
if selected_db:
    st.markdown(f"""
    ### 🤖 SQL 智能助手
    <span class='db-status'>当前环境: {selected_dataset_name} / {selected_db}</span>
    """, unsafe_allow_html=True)
else:
    st.title("🤖 SQL 智能助手")

# --- 欢迎页 (当没有消息时显示) ---
if not st.session_state.messages:
    st.markdown("---")
    st.markdown(f"### 👋 欢迎使用！")
    st.markdown("我可以帮你查询数据库中的信息。你可以尝试问我：")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**基础查询**\n- 查询表里有多少行数据？\n- 列出前 5 个结果。")
    with col2:
        st.info("**复杂查询**\n- 统计每个类别的平均值。\n- 连接两个表查询详细信息。")
        
    if selected_db:
        with st.expander("查看当前数据库 Schema 定义", expanded=False):
            schema_info = db_utils.get_db_schema(curr_conf["path"], selected_db, mode=curr_conf["mode"])
            st.code(schema_info, language="sql")

# --- 聊天记录渲染 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # 1. 渲染文本内容
        if "is_sql" in msg and msg["is_sql"]:
            # 如果是 SQL 消息，不再直接显示 Code，而是说明一下
            st.markdown("已生成查询语句并执行：")
        else:
            st.markdown(msg["content"])

        # 2. 渲染 SQL 和 结果 (使用 Tabs 优化布局)
        if "is_sql" in msg and msg["is_sql"]:
            tab_code, tab_data = st.tabs(["🧠 SQL 代码", "📊 执行结果"])
            
            with tab_code:
                st.code(msg["content"], language="sql")
                
            with tab_data:
                if "error" in msg and msg["error"]:
                    st.error(f"执行出错: {msg['error']}")
                elif "dataframe" in msg and msg["dataframe"] is not None:
                    # 使用 container width 让表格铺满
                    st.dataframe(msg["dataframe"], use_container_width=True)
                else:
                    st.info("查询执行成功，但结果为空。")

# --- 输入处理逻辑 ---
if prompt := st.chat_input("在此输入你的业务问题..."):
    if not selected_db:
        st.error("⚠️ 请先在左侧侧边栏选择一个数据库！")
    else:
        # 1. 用户消息上屏
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. 模型推理
        with st.chat_message("assistant"):
            # 创建占位符，用于显示进度
            status_container = st.empty()
            
            try:
                # 步骤 A: 分析 Schema
                status_container.markdown("🔄 *正在分析数据库结构...*")
                current_conf = settings.DATASET_CONFIG[st.session_state.selected_dataset]
                schema = db_utils.get_db_schema(current_conf["path"], selected_db, mode=current_conf["mode"])
                
                # 步骤 B: 生成 SQL
                status_container.markdown("🧠 *正在构建 SQL 逻辑...*")
                generated_text = model_utils.generate_sql_query(model, tokenizer, prompt, schema)
                
                # 清除状态文字
                status_container.empty()

                # 步骤 C: 结果处理
                if is_sql_statement(generated_text):
                    st.markdown("已生成查询语句并执行：")
                    
                    # 创建 Tabs
                    tab_code, tab_data = st.tabs(["🧠 SQL 代码", "📊 执行结果"])
                    
                    with tab_code:
                        st.code(generated_text, language="sql")
                    
                    with tab_data:
                        with st.spinner("正在数据库中检索数据..."):
                            df_result, error = db_utils.execute_sql(
                                current_conf["path"], 
                                selected_db, 
                                generated_text, 
                                mode=current_conf["mode"]
                            )
                            
                        if error:
                            st.error(f"Error: {error}")
                            df_to_save = None
                        else:
                            st.dataframe(df_result, use_container_width=True)
                            df_to_save = df_result
                            
                    new_msg = {
                        "role": "assistant",
                        "content": generated_text,
                        "is_sql": True,
                        "dataframe": df_to_save,
                        "error": error
                    }
                else:
                    # 闲聊模式
                    st.markdown(generated_text)
                    new_msg = {
                        "role": "assistant",
                        "content": generated_text,
                        "is_sql": False,
                        "dataframe": None,
                        "error": None
                    }

                # 3. 保存记录
                st.session_state.messages.append(new_msg)
                history_utils.save_conversation(
                    st.session_state.current_chat_id, 
                    st.session_state.messages,
                    st.session_state.selected_dataset,
                    selected_db
                )
                
                # 强制刷新以更新侧边栏历史记录标题
                time.sleep(0.1) 
                st.rerun()

            except Exception as e:
                status_container.empty()
                st.error(f"系统发生异常: {str(e)}")