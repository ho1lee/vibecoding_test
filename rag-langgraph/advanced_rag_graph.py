"""
고급 기능을 포함한 LangGraph RAG 시스템
- 다중 검색 전략
- 재순위화 (Reranking)
- 하이브리드 검색
- 메타데이터 필터링
"""

import os
from typing import TypedDict, Annotated, Sequence, List, Dict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from dotenv import load_dotenv
import operator
from datetime import datetime
import json

# 환경 변수 로드
load_dotenv()

# State 정의
class AdvancedAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    context: str
    answer: str
    sources: List[str]
    search_type: str  # "similarity", "mmr", "hybrid"
    metadata_filter: Optional[Dict]
    rerank_scores: List[float]
    retrieved_docs: List[Document]

class AdvancedRAGGraph:
    def __init__(self, persist_directory="./advanced_chroma_db"):
        """고급 RAG Graph 초기화"""
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview")
        self.reranker_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # Vector Store 초기화
        if os.path.exists(persist_directory):
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
        else:
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
        
        # Graph 구성
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """고급 LangGraph 워크플로우 구성"""
        workflow = StateGraph(AdvancedAgentState)
        
        # 노드 추가
        workflow.add_node("determine_search_strategy", self.determine_search_strategy)
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("rerank", self.rerank_documents)
        workflow.add_node("generate", self.generate_answer)
        workflow.add_node("validate_answer", self.validate_answer)
        workflow.add_node("format_output", self.format_output)
        
        # 조건부 엣지 추가
        workflow.set_entry_point("determine_search_strategy")
        workflow.add_edge("determine_search_strategy", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", "validate_answer")
        
        # 검증 결과에 따른 분기
        workflow.add_conditional_edges(
            "validate_answer",
            self.should_regenerate,
            {
                "regenerate": "generate",
                "continue": "format_output"
            }
        )
        
        workflow.add_edge("format_output", END)
        
        # 컴파일
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def determine_search_strategy(self, state: AdvancedAgentState) -> dict:
        """쿼리 분석을 통한 검색 전략 결정"""
        query = state["query"]
        
        # 쿼리 분석 프롬프트
        analysis_prompt = ChatPromptTemplate.from_template("""
        다음 쿼리를 분석하여 가장 적합한 검색 전략을 결정하세요.
        
        쿼리: {query}
        
        검색 전략 옵션:
        - similarity: 일반적인 유사도 검색 (기본)
        - mmr: 다양성을 고려한 검색 (Maximum Marginal Relevance)
        - hybrid: 키워드와 의미 검색을 결합
        
        응답은 다음 JSON 형식으로 해주세요:
        {{
            "search_type": "similarity|mmr|hybrid",
            "reasoning": "선택 이유"
        }}
        """)
        
        chain = analysis_prompt | self.reranker_llm | StrOutputParser()
        result = chain.invoke({"query": query})
        
        try:
            analysis = json.loads(result)
            search_type = analysis.get("search_type", "similarity")
        except:
            search_type = "similarity"
        
        return {"search_type": search_type}
    
    def retrieve_documents(self, state: AdvancedAgentState) -> dict:
        """검색 전략에 따른 문서 검색"""
        query = state["query"]
        search_type = state.get("search_type", "similarity")
        metadata_filter = state.get("metadata_filter", {})
        
        # 검색 수행
        if search_type == "mmr":
            # Maximum Marginal Relevance 검색
            docs = self.vectorstore.max_marginal_relevance_search(
                query, 
                k=6,
                filter=metadata_filter if metadata_filter else None
            )
        elif search_type == "hybrid":
            # 하이브리드 검색 (시뮬레이션)
            # 실제로는 BM25 + 벡터 검색을 결합
            similarity_docs = self.vectorstore.similarity_search(
                query, 
                k=3,
                filter=metadata_filter if metadata_filter else None
            )
            mmr_docs = self.vectorstore.max_marginal_relevance_search(
                query, 
                k=3,
                filter=metadata_filter if metadata_filter else None
            )
            # 중복 제거
            seen = set()
            docs = []
            for doc in similarity_docs + mmr_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    docs.append(doc)
        else:
            # 기본 유사도 검색
            docs = self.vectorstore.similarity_search(
                query, 
                k=6,
                filter=metadata_filter if metadata_filter else None
            )
        
        return {"retrieved_docs": docs}
    
    def rerank_documents(self, state: AdvancedAgentState) -> dict:
        """LLM을 사용한 문서 재순위화"""
        query = state["query"]
        docs = state["retrieved_docs"]
        
        if not docs:
            return {
                "context": "",
                "sources": [],
                "rerank_scores": []
            }
        
        # 재순위화 프롬프트
        rerank_prompt = ChatPromptTemplate.from_template("""
        다음 쿼리와 문서의 관련성을 0-10 점수로 평가하세요.
        
        쿼리: {query}
        
        문서: {document}
        
        평가 기준:
        - 쿼리와의 직접적인 관련성
        - 정보의 구체성과 유용성
        - 답변에 필요한 핵심 정보 포함 여부
        
        점수만 숫자로 응답하세요.
        """)
        
        chain = rerank_prompt | self.reranker_llm | StrOutputParser()
        
        # 각 문서에 대한 점수 계산
        scored_docs = []
        for doc in docs:
            try:
                score = float(chain.invoke({
                    "query": query,
                    "document": doc.page_content[:500]  # 처음 500자만 평가
                }))
            except:
                score = 5.0  # 기본 점수
            
            scored_docs.append((score, doc))
        
        # 점수 기준으로 정렬
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # 상위 4개 문서 선택
        top_docs = scored_docs[:4]
        rerank_scores = [score for score, _ in top_docs]
        selected_docs = [doc for _, doc in top_docs]
        
        # 컨텍스트 생성
        context = "\n\n".join([doc.page_content for doc in selected_docs])
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in selected_docs]))
        
        return {
            "context": context,
            "sources": sources,
            "rerank_scores": rerank_scores,
            "messages": [HumanMessage(content=f"Query: {query}\nContext: {context}")]
        }
    
    def generate_answer(self, state: AdvancedAgentState) -> dict:
        """향상된 답변 생성"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 주어진 컨텍스트를 기반으로 정확하고 도움이 되는 답변을 제공하는 AI 어시스턴트입니다.
            
            다음 지침을 따르세요:
            1. 컨텍스트에 있는 정보만을 사용하여 답변하세요.
            2. 답변할 수 없는 경우, 명확히 그 사실을 알리세요.
            3. 가능한 한 구체적이고 상세한 답변을 제공하세요.
            4. 필요한 경우 단계별 설명이나 예시를 포함하세요.
            5. 정보의 출처나 신뢰도에 대해 언급할 수 있다면 포함하세요.
            """),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"messages": state["messages"]})
        
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)]
        }
    
    def validate_answer(self, state: AdvancedAgentState) -> dict:
        """답변 품질 검증"""
        answer = state["answer"]
        query = state["query"]
        
        validation_prompt = ChatPromptTemplate.from_template("""
        다음 질문과 답변을 평가하세요:
        
        질문: {query}
        답변: {answer}
        
        평가 기준:
        1. 답변이 질문에 직접적으로 대답하는가?
        2. 답변이 충분히 구체적이고 유용한가?
        3. 답변에 모순이나 오류가 없는가?
        
        응답 형식:
        {{
            "is_valid": true/false,
            "score": 0-10,
            "issues": ["문제점1", "문제점2"]
        }}
        """)
        
        chain = validation_prompt | self.reranker_llm | StrOutputParser()
        
        try:
            validation = json.loads(chain.invoke({
                "query": query,
                "answer": answer
            }))
            
            # 점수가 7점 미만이면 재생성
            if validation.get("score", 10) < 7:
                return {"validation_result": "regenerate"}
        except:
            pass
        
        return {"validation_result": "continue"}
    
    def should_regenerate(self, state: AdvancedAgentState) -> str:
        """재생성 여부 결정"""
        return state.get("validation_result", "continue")
    
    def format_output(self, state: AdvancedAgentState) -> dict:
        """최종 출력 포맷팅"""
        return {
            "messages": state["messages"],
            "query": state["query"],
            "answer": state["answer"],
            "sources": state["sources"],
            "context": state["context"],
            "search_type": state.get("search_type", "similarity"),
            "rerank_scores": state.get("rerank_scores", [])
        }
    
    def add_documents_from_directory(self, directory_path: str, glob_pattern: str = "**/*"):
        """디렉토리에서 문서 일괄 추가"""
        loader = DirectoryLoader(
            directory_path,
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        
        documents = loader.load()
        
        # 메타데이터 추가
        for doc in documents:
            doc.metadata["indexed_at"] = datetime.now().isoformat()
            doc.metadata["file_type"] = os.path.splitext(doc.metadata["source"])[1]
        
        # 문서 분할
        split_docs = self.text_splitter.split_documents(documents)
        
        if split_docs:
            self.vectorstore.add_documents(split_docs)
            print(f"{len(split_docs)}개의 문서 청크가 추가되었습니다.")
    
    def query_with_metadata(self, question: str, metadata_filter: Dict = None, 
                           search_type: str = "similarity", thread_id: str = "default") -> dict:
        """메타데이터 필터와 검색 전략을 지정한 쿼리"""
        initial_state = {
            "messages": [],
            "query": question,
            "context": "",
            "answer": "",
            "sources": [],
            "search_type": search_type,
            "metadata_filter": metadata_filter,
            "rerank_scores": [],
            "retrieved_docs": []
        }
        
        # Graph 실행
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(initial_state, config)
        
        return result
    
    def get_statistics(self) -> dict:
        """벡터 스토어 통계 정보"""
        collection = self.vectorstore._collection
        count = collection.count()
        
        return {
            "total_documents": count,
            "persist_directory": self.persist_directory,
            "embedding_model": "OpenAI",
            "vector_store": "Chroma"
        }
