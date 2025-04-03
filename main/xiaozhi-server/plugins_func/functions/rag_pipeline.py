from plugins_func.register import register_function,ToolType, ActionResponse, Action
from config.logger import setup_logging
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever, \
    ElasticsearchEmbeddingRetriever
from haystack.components.joiners import DocumentJoiner
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.utils import Secret
import os

TAG = __name__
logger = setup_logging()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ES_HOST = "https://10.177.44.113:9200/"
ES_USER = "elastic"
ES_PASSWORD = "password_from_yulu"

rag_pipeline_function_desc = {
                "type": "function",
                "function": {
                    "name": "rag_pipeline",
                    "description": "当用户想使用检索增强生成(RAG)时调用",
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

class RAGSystem:
    def __init__(self):
        self.document_store = ElasticsearchDocumentStore(
            hosts=ES_HOST,
            basic_auth=(ES_USER, ES_PASSWORD),
            # ca_certs="/Users/tangyulu/http_ca.crt",
            verify_certs=False,
            index="wechat_chunks"
        )

        # 初始化组件
        self.bm25_retriever = ElasticsearchBM25Retriever(document_store=self.document_store, top_k=5)
        self.embedding_retriever = ElasticsearchEmbeddingRetriever(
            document_store=self.document_store,
            top_k=5
        )
        self.query_embedder = SentenceTransformersTextEmbedder(
            model="/your_path/distiluse-base-multilingual-cased-v2"
        )
        self.joiner = DocumentJoiner(weights=[0.6, 0.4])  # 混合权重
        self.prompt_builder = PromptBuilder(
            template='''基于以下背景信息回答问题。如果信息不足，请说明无法回答。

                    背景信息：
                    {% for doc in documents %}{{ doc.content }}{% endfor %}

                    问题：{{ question }}
                    要求：用中文回答，保持简洁专业
                    '''
        )
        self.generator = OpenAIGenerator(
                    api_key=Secret.from_token("your_api_key"),
                    api_base_url="your_api_base_url",
                    model="your_model"
                )
        self.answer_builder = AnswerBuilder()

        # 构建管道
        self.pipeline = Pipeline()
        self._connect_components()

    def _connect_components(self):
        self.pipeline.add_component("query_embedder", self.query_embedder)
        self.pipeline.add_component("bm25_retriever", self.bm25_retriever)
        self.pipeline.add_component("embedding_retriever", self.embedding_retriever)
        self.pipeline.add_component("joiner", self.joiner)
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("generator", self.generator)
        self.pipeline.add_component("answer_builder", self.answer_builder)

        self.pipeline.connect("query_embedder.embedding", "embedding_retriever.query_embedding")
        self.pipeline.connect("bm25_retriever.documents", "joiner.documents")
        self.pipeline.connect("embedding_retriever.documents", "joiner.documents")
        self.pipeline.connect("joiner.documents", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder.prompt", "generator.prompt")
        self.pipeline.connect("generator.replies", "answer_builder.replies")
        self.pipeline.connect("joiner.documents", "answer_builder.documents")

    def set_streaming_callback(self, callback):
        """允许外部设置流式输出的回调函数"""
        self.generator.streaming_callback = callback

    def run(self, question: str, top_k: int = 5):
        try:
            # 运行管道
            logger.bind(tag=TAG).info(f"开始运行pipeline,question:{question}")
            results = self.pipeline.run({
                "query_embedder": {"text": question},
                "bm25_retriever": {"query": question},
                "embedding_retriever": {"top_k": top_k},
                "prompt_builder": {"question": question},
                "answer_builder": {"query": question}
            })
            logger.bind(tag=TAG).info(f"pipeline完成")
            if results["answer_builder"]["answers"]:
                return results["answer_builder"]["answers"][0].data
            else:
                return "无法找到相关信息"
        except Exception as e:
            return f"发生错误：{str(e)}"
def streaming_callback(token):
    logger.bind(tag=TAG).info(f"流式输出: {token}")
    return token
@register_function('rag_pipeline', rag_pipeline_function_desc, ToolType.WAIT)
def rag_pipeline(question: str):
    logger.bind(tag=TAG).info(f"RAG开始处理问题: {question}")
    rag_system = RAGSystem()   
    logger.bind(tag=TAG).info(f"初始化完成")
    rag_res = rag_system.run(question)
    # res=rag_system.set_streaming_callback(streaming_callback)
    return ActionResponse(action=Action.RESPONSE, result="RAG已处理", response=rag_res)
