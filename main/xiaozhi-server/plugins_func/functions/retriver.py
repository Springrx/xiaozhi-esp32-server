from haystack import Document
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from elasticsearch import Elasticsearch
from plugins_func.register import register_function,ToolType, ActionResponse, Action
rag_pipeline_function_desc = {
                "type": "function",
                "function": {
                    "name": "rag_pipeline",
                    "description": "当用户询问复旦大学相关的内容时使用",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "用户的问题"
                            }
                        },
                        "required": ["question"]
                    }
                }
            }

ES_HOST= ""
ES_USER= "elastic"
ES_PASSWORD= ""

document_store = ElasticsearchDocumentStore(            
    hosts=ES_HOST,
    basic_auth=(ES_USER, ES_PASSWORD),
    verify_certs=False,
    index="wechat_chunks"
)
def retrieve(query:str):
    retriever = ElasticsearchBM25Retriever(document_store=document_store, top_k=5)
    res=retriever.run(query=query)
    return res
@register_function('rag_pipeline', rag_pipeline_function_desc, ToolType.WAIT)
def rag_pipeline(question: str):
    ret=retrieve(question)
    retriever_content=''
    for doc in ret["documents"]:
        retriever_content+=f"{doc.content}\n\n"
    print(doc.content)
    rag_prompt = (
            f"根据下列数据，用中文回应用户的请求：{question}\n\n"
            f"数据为：{retriever_content}"
            f"由于文档比较杂乱，你可以首先寻找有效信息，然后根据有效信息作答，如果不能确定答案，告知用户可以说'这个我还不太清楚'"
        )
    return ActionResponse(action=Action.REQLLM, result=rag_prompt, response=None)
