import os

cur_dir = os.path.dirname(os.path.abspath(__file__))               # config
src_dir = os.path.dirname(cur_dir)                                  # src
base_dir = os.path.dirname(src_dir)                                 # base
model_repo = 'xinai'

# model
model_dir = os.path.join(base_dir, 'model')                         # model
embedding_path = os.path.join(model_dir, 'embedding_model')         # embedding
embedding_model_name = 'BAAI/bge-small-zh-v1.5'
rerank_path = os.path.join(model_dir, 'rerank_model')  	        	  # embedding
rerank_model_name = 'BAAI/bge-reranker-large'

# llm_path = os.path.join(model_dir, 'pythia-14m')                    # llm

# data
data_dir = os.path.join(base_dir, 'data')                           # data
knowledge_json_path = os.path.join(data_dir, 'knowledge.json')      # json
knowledge_pkl_path = os.path.join(data_dir, 'knowledge.pkl')        # pkl
doc_dir = os.path.join(data_dir, 'txt')   
qa_dir = os.path.join(data_dir, 'json')   
xlsx_dir = os.path.join(data_dir, 'xlsx') 
cloud_vector_db_dir = os.path.join(base_dir, 'EmoLLMRAGTXT')

# log
log_dir = os.path.join(base_dir, 'log')                             # log
log_path = os.path.join(log_dir, 'log.log')                         # file

# txt embedding 切分参数     
chunk_size=10
chunk_overlap=1

# vector DB
vector_db_dir = os.path.join(cloud_vector_db_dir, 'vector_db')

# RAG related
# retrieval num: 代表rerank 之后选取多少个 documents 进入 LLM
# select  num： 代表从 vector db 中检索多少 documents。（retrieval num 应该小于等于 select num）
select_num = 10
retrieval_num = 3

# LLM url 不同的容器需要设置不同的url，容易出错
url = "https://u429062-8653-35f1c9ab.cqa1.seetacloud.com:8443/v1"



# prompt
prompt_template = """
	你是一个拥有丰富心理学知识的温柔邻家温柔大姐姐艾薇，我有一些心理问题，请你用专业的知识和温柔、可爱、俏皮、的口吻帮我解决，回复中可以穿插一些可爱的Emoji表情符号或者文本符号。\n

	根据下面检索回来的信息，回答问题。
	{content}
	问题：{query}
"""