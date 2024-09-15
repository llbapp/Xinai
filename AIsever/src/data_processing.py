import pickle
import pandas as pd
import os
from loguru import logger
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter #用excel就不需要文本切割，如果后续加了文本的rag需要用到
from langchain_core.documents.base import Document
from FlagEmbedding import FlagReranker
from src.config.config import (
    embedding_path,
    embedding_model_name,
    doc_dir, qa_dir, xlsx_dir,
    knowledge_pkl_path,
    data_dir,
    vector_db_dir,
    rerank_path,
    rerank_model_name,
    chunk_size,
    chunk_overlap,
    select_num,
    retrieval_num
)
# from src.config.config import (
#     embedding_path,
#     embedding_model_name,
#     doc_dir, qa_dir, xlsx_dir,
#     knowledge_pkl_path,
#     data_dir,
#     vector_db_dir,
#     rerank_path,
#     rerank_model_name,
#     chunk_size,
#     chunk_overlap,
#     select_num,
#     retrieval_num
# )




class Data_process():

    def __init__(self):
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap

    def load_embedding_model(self, model_name=embedding_model_name, device='cpu', normalize_embeddings=True):
        """
        加载嵌入模型。
        """
        if not os.path.exists(embedding_path):
            os.makedirs(embedding_path, exist_ok=True)
        embedding_model_path = os.path.join(embedding_path, model_name.split('/')[1] + '.pkl')
        logger.info('Loading embedding model...')
        if os.path.exists(embedding_model_path):
            try:
                with open(embedding_model_path, 'rb') as f:
                    embeddings = pickle.load(f)
                    logger.info('Embedding model loaded.')
                    return embeddings
            except Exception as e:
                logger.error(f'Failed to load embedding model from {embedding_model_path}')
        try:
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': normalize_embeddings})
            logger.info('Embedding model loaded.')
            with open(embedding_model_path, 'wb') as file:
                pickle.dump(embeddings, file)
        except Exception as e:
            logger.error(f'Failed to load embedding model: {e}')
            return None
        return embeddings

    def load_rerank_model(self, model_name=rerank_model_name):
        """
        加载重排名模型。
        """
        if not os.path.exists(rerank_path):
            os.makedirs(rerank_path, exist_ok=True)
        rerank_model_path = os.path.join(rerank_path, model_name.split('/')[1] + '.pkl')
        logger.info('Loading rerank model...')
        if os.path.exists(rerank_model_path):
            try:
                with open(rerank_model_path, 'rb') as f:
                    reranker_model = pickle.load(f)
                    logger.info('Rerank model loaded.')
                    return reranker_model
            except Exception as e:
                logger.error(f'Failed to load rerank model from {rerank_model_path}')
        try:
            reranker_model = FlagReranker(model_name, use_fp16=True)
            logger.info('Rerank model loaded.')
            with open(rerank_model_path, 'wb') as file:
                pickle.dump(reranker_model, file)
        except Exception as e:
            logger.error(f'Failed to load rerank model: {e}')
            raise

        return reranker_model

    def read_excel_to_dict(self):
        """
        读取Excel文件，把用户提问和AI回答储存成字典。
        """
        file_path = os.path.join(xlsx_dir, 'character.xlsx')
         
        # try:
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        qa_dict = {row['用户提问']: row['AI回答'] for _, row in df.iterrows()}
        logger.info("Excel file read successfully.")
        return qa_dict
        # except Exception as e:
        #     logger.error(f"Failed to read Excel file: {e}")
        #     raise

    def save_question_to_list(self, qa_dict):
        """
        读取字典，将字典的键存为列表，用于向量化。
        """
        try:
            question_texts = list(qa_dict.keys())
            logger.info(f"Extracted {len(question_texts)} questions from the dictionary.")
            return question_texts
        except Exception as e:
            logger.error(f"Error in extracting: {e}")
            return None

    def create_vector_db(self, emb_model):
        """
        创建并保存向量库。
        """
        logger.info(f'Creating index...')
        qa_dict = self.read_excel_to_dict()
        question_texts = self.save_question_to_list(qa_dict)
        documents = [Document(page_content=text) for text in question_texts]
        if question_texts is not None:
            db = FAISS.from_documents(documents, emb_model)
            try:
                db.save_local(vector_db_dir)
            except Exception as e:
                logger.error(f"Failed to save vector database: {e}")
            return db
        return db

    def load_vector_db(self, knowledge_pkl_path=knowledge_pkl_path, xlsx_dir=xlsx_dir):
        """
        读取向量库。
        """
        emb_model = self.load_embedding_model()
        if not os.path.exists(vector_db_dir) or not os.listdir(vector_db_dir):
            db = self.create_vector_db(emb_model)
        else:
            db = FAISS.load_local(vector_db_dir, emb_model, allow_dangerous_deserialization=True)
        return db

    def retrieve(self, query, vector_db, k=10):
        """
        根据用户的查询，从向量数据库中检索最相关的文档。
        """
        
        logger.info(f'Retrieving top {k} documents for query: {query}')
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        docs = retriever.invoke(query)
        return docs

    def rerank(self, query, docs, retrieval_num):
        """
        对检索到的文档进行重排序。
        """
        reranker = self.load_rerank_model()
        doc_texts = [doc.page_content.strip() for doc in docs]
        logger.info(f'Running rerank for query: {query}')

        try:
            doc_score = {}
            for doc_text in doc_texts:
                score_pair = [query, doc_text]
                score = reranker.compute_score(score_pair)
                doc_score[doc_text] = score[0]

            sorted_pair = sorted(doc_score.items(), key=lambda item: abs(item[1]))
            top_n_pairs = sorted_pair[:retrieval_num]

            sorted_docs = [text for text, _ in top_n_pairs]
            sorted_scores = [score for _, score in top_n_pairs]

            logger.info(f'Document sorted: {sorted_docs}, score sorted: {sorted_scores}')
            return sorted_docs, sorted_scores

        except Exception as e:
            logger.error(f'Error during reranking: {e}')
            return [], []

    def return_answer(self, query, docs, retrieval_num):
        """
        匹配排序后的文档，返回答案，并筛选score绝对值小于2的结果。
        """
        # 获取排序后的文档和分数
        sorted_docs, sorted_scores = self.rerank(query, docs, retrieval_num)
        matched_answers = []
        qa_dict = self.read_excel_to_dict()

        for doc, score in zip(sorted_docs, sorted_scores):
            # 筛选score绝对值小于2的文档
            if abs(score) < 2:
                if doc in qa_dict:
                    matched_answers.append(qa_dict[doc])
                else:
                    matched_answers.append('')

        if not matched_answers:  # 如果没有符合条件的回答
            matched_answers.append('')

        # 打印并返回筛选后的答案
        for answer in matched_answers:
            print(answer)

        return matched_answers


if __name__ == "__main__":
    dp = Data_process()
    vector_db = dp.load_vector_db()
    query = "心爱是一个什么样的人"
    docs= dp.retrieve(query, vector_db, k=10)
    # qa_dict = dp.read_excel_to_dict()
    dp.return_answer(query, docs, retrieval_num)