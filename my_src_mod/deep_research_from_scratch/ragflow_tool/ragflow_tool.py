import os
import asyncio
import traceback
from typing import Annotated, List, Optional
from dataclasses import dataclass

from ragflow_sdk import RAGFlow

from langchain_core.tools import InjectedToolArg, tool

@dataclass
class RetrievalResult:
    """检索结果数据类"""
    query: str
    document_name: str
    document_id: str
    content: str
    similarity: float
    dataset_id: str
    dataset_names: str
    
    def to_dict(self):
        return {
            "query": self.query,
            "document_name": self.document_name,
            "document_id": self.document_id,
            "content": self.content,
            "similarity": self.similarity,
            "dataset_id": self.dataset_id,
            "dataset_names": self.dataset_names

        }

_client = None
def get_ragflow_client():
    if _client is None:
        _client = RAGFlow(
            api_key=os.getenv("RAGFLOW_API_KEY"), 
            base_url=os.getenv("RAGFLOW_API_URL")
        )
    return _client

@tool(parse_docstring=True)
async def knowledge_search(
    queries: List[str]
) -> str:
    """Retrieve relevant chunks from the knowledge database retrieve interface based on the question.

    Args:
        queries: A list of search queries to execute

    Returns:
        Formatted string of search results
    """
    # 单次搜索
    async def ragflow_search_single(
        query: str,
        dataset_names: List[str],
        page_size: int,
        similarity_threshold: float,
        vector_similarity_weight: float
    ) -> str:
        try:
            client = get_ragflow_client()
            chunks = client.retrieve(
                dataset_names=dataset_names,
                page_size=page_size,
                similarity_threshold=similarity_threshold,
                vector_similarity_weight=vector_similarity_weight
            )
            # 获取document_name
            docid_name = dict()
            for dataset in dataset_names:
                dataset_instance = client.get_dataset(dataset)
                for doc in dataset_instance.list_documents():
                    docid_name[doc.id] = doc.name
            # 获取检索结果
            results = []
            for chunk in chunks:
                result = RetrievalResult(
                    query=query,
                    document_name=getattr(chunk, 'document_name', 'Unknown'),
                    document_id=docid_name[getattr(chunk, 'document_id', '')],
                    content=getattr(chunk, 'content', ''),
                    similarity=float(getattr(chunk, 'similarity', 0.0) or 0.0),
                    dataset_id=getattr(chunk, 'dataset_id', ''),
                    dataset_names=dataset_names
                )
                results.append(result)
            results.sort(key=lambda x: x.similarity, reverse=True)
            return query, results, None
        except Exception as e:
            return query, [], str(e)
    
    coros = [
        ragflow_search_single(
            query, 
            os.getenv("RAGFLOW_DATASET_NAMES"), 
            os.getenv("RAGFLOW_PAGE_SIZE"), 
            os.getenv("RAGFLOW_SIMILARITY_THRESHOLD"), 
            os.getenv("RAGFLOW_VECTOR_SIMILARITY_WEIGHT")
        ) for query in queries
    ]
    search_results = await asyncio.gather(*coros)

    # 格式化输出
    lines: List[str] = ["知识搜索结果:"]
    for query, results, err in search_results:
        if err or not results:
            continue
        lines.append("")
        lines.append(f"--- 查询: {query} ---")
        # 截断结果
        results = results[:os.getenv("RAGFLOW_PAGE_SIZE")]
        # 拼接结果
        for i, result in enumerate(results, 1):
            excerpt = result.content.replace("\n", " ")[:300]
            title = result.document_name
            lines.append(f"[{i}] {title}")
            lines.append(f"相似度: {result.similarity:.4f}")
            lines.append(excerpt)
        # 来源
        sources = set()
        for result in results:
            sources.update(result.dataset_names)
        lines.append(f"来源: {", ".join(sources)}")
    return "\n".join(lines)