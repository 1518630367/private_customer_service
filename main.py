import torch
import numpy as np
from  config import  config
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import  random
#连接服务器
connections.connect(host="localhost", port="19530")

fields = [
    FieldSchema(name="ids", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="question", dtype=DataType.VARCHAR,max_length = config.max_question_len),
    FieldSchema(name="answer", dtype=DataType.VARCHAR,max_length = config.max_answer_len),
    FieldSchema(name="state", dtype=DataType.INT64,max_length = 512),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=config.embedding_dim)
]
schema = CollectionSchema(fields, "建立数据库连接")
# #连接数据库
my_milvus = Collection("qa_system",schema)


#创建数据
entity = [
    ["Is this milvus ?","Is this milvus ?"],
    ["yes","no"],
    [0,0],
    [[random.random() for _ in range(384)] for _ in range(2)]
]
#输入数据

insert_result = my_milvus.insert(entity)

#释放缓存
my_milvus.flush()

index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}
my_milvus.create_index("embeddings", index)

my_milvus.load()
vectors_to_search = entity[-1][-1:]
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}
result = my_milvus.search(vectors_to_search, "embeddings", search_params, limit=1, output_fields=["answer"])
print(result)
connections.disconnect(alias="default")







