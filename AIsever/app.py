import streamlit as st
from src.pipeline import Chatchain
from loguru import logger
from src.data_processing import Data_process
from src.config.config import select_num

chain = Chatchain()
dp = Data_process()
vector_db = dp.load_vector_db()
qa_list = []

st.title("心爱demo")

# 初始化history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query" not in st.session_state:
    st.session_state.query = ""  # 保存用户的输入

# 展示对话
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg["content"])

# 获取用户输入
if query := st.chat_input("Say something"):
    # 保存用户输入到 session_state
    st.session_state.query = query
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(query)
    # 保存用户消息到会话历史
    st.session_state.messages.append({"role": "user", "content": query})

# 检查是否有用户输入并执行后续逻辑
if st.session_state.query:
    # 获取检索结果
    docs = dp.retrieve(st.session_state.query, vector_db, k=10)
    # 获取生成的答案
    content = dp.return_answer(st.session_state.query, docs, select_num)
    # 创建对话 session 并获取响应
    response = chain.create_chat_session(st.session_state.query, content)
    
    # 显示助手的回复
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # 保存助手回复到会话历史
    st.session_state.messages.append({"role": "assistant", "content": response})

