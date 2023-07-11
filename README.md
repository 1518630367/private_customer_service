# 项目说明
    智能客服机器人，根据用户提出的问题以及本地维护的答案库中的答案，来回答用户的相关问题。
### 业务背景
    用于回答常见的业务问题
### 实现逻辑
    整体的实现逻辑为 
        1.构建向量数据库，将已经知道的答案转化为向量，储存在向量数据库之中
        2.将用户提出的问题转化为向量，之后从向量数据库中检索相似度最大的topk个候选答案
        3.把topk个候选答案以及用户提出的问题输入Chatgpt，让它总结生成合适的答案，更加顺滑的反馈给用户
    实现细节：
        1.选择的向量数据库为Milvus(Milvus 为开源向量数据库，它改进了 Faiss 和 hnswlib 等高性能存储和索引库，保证了时间和资源高效的查询速度。可在毫秒级内检索万亿级数据集上的矢量数据。)
        2.选择的向量转化模型为paraphrase-multilingual-MiniLM-L12-v2（支持多语言），并且为了更好的匹配在将答案转化向量时还拼接了答案的关键词
        具体为: "答案关键词"[SEP]"答案"，然后输入模型转化为384维度的向量
        3.在搜索答案时，把问题转化为384的向量，根据余弦相似度来检索余弦相似度最大的答案向量，返回余弦相似度最大的topk个问题
        4.调用Chatgpt的api为："config/api_key",当答案库没有可以回答问题的答案时，返回指定话术
        5. question_answer_pair.csv为待加入的新的答案，用于插入Milvus数据库
        6.convert_to_vecor.py用于维护数据库，具体为把question_answer_pair.csv的数据转化为向量输入数据库
        7.customer_service.py为项目逻辑，用于回答问题
### 模型说明：
    algorithm/pretrained_models/customer_service/paraphrase-multilingual-MiniLM-L12-v2 （用于转化句子为向量）