from openai import OpenAI
from src.data_processing import Data_process
from src.config.config import (
    select_num
)
from src.util.llm import load_llm
# url = "https://u429062-8653-35f1c9ab.cqa1.seetacloud.com:8443/v1"


class Chatchain():
    def __init__(self):
        self.chat_history = []  # 初始化时保留对话历史

    def create_chat_session(self, message, content):
        openai, model = load_llm()
        max_tokens = 100

        # 系统消息，只在第一次对话时添加
        if len(self.chat_history) == 0:
            system_input = {
                "role": "system",
                "content": f"你是一个拥有丰富心理学知识的温柔邻家温柔女大学生心爱，我有一些心理问题，\
                请你用专业的知识和温柔、可爱、俏皮的口吻帮我解决，\
                回复中可以穿插一些可爱的Emoji表情符号或者文本符号。\n根据下面检索回来的信息，回答问题。{content}"
            }
            self.chat_history.append(system_input)

        # 用户输入
        user_input = {
            "role": "user",
            "content": message
        }
        self.chat_history.append(user_input)

        # 调用 OpenAI API 获取回复
        response = openai.chat.completions.create(
            model=model,
            messages=self.chat_history,
            max_tokens=max_tokens
        )

        # 助手回复
        reply = response.choices[0].message.content
        assistant_input = {
            "role": "assistant",
            "content": reply
        }
        self.chat_history.append(assistant_input)

        return reply


if __name__ == "__main__":
    chain = Chatchain()
    dp = Data_process()
    vector_db = dp.load_vector_db()
    
    while True:
        query = input("请输入：（输入exit退出对话）")
        if query == "exit":
            break

        docs, retriever = dp.retrieve(query, vector_db, k=10)
        content = dp.return_answer(query, docs, select_num)
        response = chain.create_chat_session(query, content)
        print(f'心爱：{response}')
