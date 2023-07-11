from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query
import json
from  config import  config
import openai
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import uvicorn
app = FastAPI()
openai.api_key = config.api_key
# 调用chatgpt
def chatgpt(content):
    messages = []
    messages.append({"role": "user", "content": '"' + content + '"'})
    model = "gpt-3.5-turbo-0301"
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=800,
        )
        res = response['choices'][0]['message']['content']
        return res
    except Exception as e:
        print(repr(e))
        return "chatgpt 响应异常"

def get_question_embedding(model,question):
    embeddings = model.encode(question)
    return embeddings
def get_ans(question_embeddings):
    # 连接服务
    connections.connect(host="localhost", port="19530")
    # #连接数据库
    my_milvus = Collection("qa_system")
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    my_milvus.create_index("embeddings", index)
    my_milvus.load()
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    result = my_milvus.search(question_embeddings, "embeddings", search_params, limit=config.topk, output_fields=["answer"])
    return  result


@app.get('/customer_service')
async def main(question = Query("")):
    model = SentenceTransformer('./customer_service/paraphrase-multilingual-MiniLM-L12-v2')
    question_embedding = get_question_embedding(model,[question])
    ans = get_ans(question_embedding)
    dic = dict()
    answer = ""
    for hits in ans:
        for hit in hits:
            answer = hit.entity.get("answer")
    no_answer = "Sorry,I can't answer your question,please wait for the manual customer service to answer this question."
    content = f"Assuming you are a customer service representative,Someone has asked you the following question:{question},This is the approximate answer to this question{answer}," \
              f"Please help me reorganize the language and answer the customer's questions and If you believe that the given answer cannot answer the customer's question, please answer:{no_answer},Please provide only the answer and do not include any additional information and be as brief as possible."
    dic["answer"] = chatgpt(content)
    return json.loads(json.dumps(dic))

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8092)
