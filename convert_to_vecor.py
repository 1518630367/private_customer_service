from sentence_transformers import SentenceTransformer
from  config import  config
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import pandas as pd
def get_vector(model,df):
    df['sentences'] = df['question'] + " " + '[SEP]' + " " + df['answer']
    embeddings = model.encode(df['sentences'])
    df['embedding'] = embeddings.tolist()
    return df

def update_mlivus(df):
    # 连接服务
    connections.connect(host="localhost", port="19530")

    fields = [
        FieldSchema(name="ids", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=config.max_question_len),
        FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=config.max_answer_len),
        FieldSchema(name="state", dtype=DataType.INT64, max_length=512),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=config.embedding_dim)
    ]
    schema = CollectionSchema(fields, "建立数据库连接")
    # #连接数据库
    my_milvus = Collection("qa_system", schema)
    # 创建数据

    questions = df['question'].tolist()
    answers = df['answer'].tolist()
    state = [0 for _ in range(len(answers))]
    embedding = df['embedding'].tolist()
    entity = [
        questions,
        answers,
        state,
        embedding
    ]
    # 输入数据
    my_milvus.insert(entity)
    # 释放缓存
    my_milvus.flush()
if __name__ == '__main__':

    df = pd.read_csv("./question_answer_pair/question_answer_pairs.csv")
    model = SentenceTransformer('../paraphrase-multilingual-MiniLM-L12-v2')
    df = get_vector(model,df)
    update_mlivus(df)
