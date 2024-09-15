from src.pipeline import Chatchain
from loguru import logger
from src.data_processing import Data_process
from src.config.config import (
    select_num,
    retrieval_num,
)
import pandas as pd

'''
	1）构建完整的 RAG pipeline。输入为用户 query，输出为 answer
	2）调用 embedding 提供的接口对 query 向量化
	3）下载基于 FAISS 预构建的 vector DB ，并检索对应信息
	4）调用 rerank 接口重排序检索内容
	5）调用 prompt 接口获取 system prompt 和 prompt template
	6）拼接 prompt 并调用模型返回结果

'''
def main() -> str:
        chain = Chatchain()
        dp = Data_process()
        vector_db = dp.load_vector_db()
        qa_list = []
        
        while True:
            query = input("请输入：（输入exit退出对话）")
            if query == "exit":
                break

            docs = dp.retrieve(query, vector_db, k=10)
            content = dp.return_answer(query, docs, select_num)
            response = chain.create_chat_session(query, content)
            print(f'心爱：{response}')

            qa_list.append([query, response])

        df = pd.DataFrame(qa_list, columns=['用户提问', 'AI回答'])
        df.to_excel('对话记录.xlsx', index=False)  # 保存DataFrame到Excel文件
        return qa_list


if __name__ == "__main__":
    res = main()

    logger.info(res)
