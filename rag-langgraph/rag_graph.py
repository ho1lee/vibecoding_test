"""
LangGraph를 사용한 RAG (Retrieval-Augmented Generation) 시스템
"""

import os
from typing import TypedDict, Annotated, Sequence, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import Graph, StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.checkpoint import MemorySaver
from dotenv import load_dotenv
import operator

# 환경 변수 로드
load_dotenv()

# State 정의
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    context: str
    answer: str
    sources: List[str]

class RAGGraph:
    def __init__(self, persist_directory="./chroma_db"):
        """RAG Graph 초기화"""
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
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
        """LangGraph 워크플로우 구성"""
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("generate", self.generate_answer)
        workflow.add_node("format_output", self.format_output)
        
        # 엣지 추가
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "format_output")
        workflow.add_edge("format_output", END)
        
        # 컴파일
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def retrieve_documents(self, state: AgentState) -> dict:
        """관련 문서 검색"""
        query = state["query"]
        
        # 벡터 스토어에서 관련 문서 검색
        docs = self.vectorstore.similarity_search(query, k=4)
        
        # 컨텍스트 생성
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
        
        return {
            "context": context,
            "sources": sources,
            "messages": [HumanMessage(content=f"Query: {query}\nContext: {context}")]
        }
    
    def generate_answer(self, state: AgentState) -> dict:
        """LLM을 사용하여 답변 생성"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 주어진 컨텍스트를 기반으로 질문에 답변하는 도우미입니다.
            
            다음 규칙을 따르세요:
            1. 컨텍스트에 있는 정보만을 사용하여 답변하세요.
            2. 컨텍스트에 답변할 정보가 없다면, "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답변하세요.
            3. 답변은 명확하고 간결하게 작성하세요.
            4. 가능한 경우 구체적인 예시나 세부사항을 포함하세요.
            """),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"messages": state["messages"]})
        
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)]
        }
    
    def format_output(self, state: AgentState) -> dict:
        """최종 출력 포맷팅"""
        return {
            "messages": state["messages"],
            "query": state["query"],
            "answer": state["answer"],
            "sources": state["sources"],
            "context": state["context"]
        }
    
    def add_documents(self, file_paths: List[str]):
        """문서를 벡터 스토어에 추가"""
        all_documents = []
        
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                print(f"지원하지 않는 파일 형식: {file_path}")
                continue
            
            documents = loader.load()
            # 문서 분할
            split_docs = self.text_splitter.split_documents(documents)
            all_documents.extend(split_docs)
        
        if all_documents:
            # 벡터 스토어에 추가
            self.vectorstore.add_documents(all_documents)
            print(f"{len(all_documents)}개의 문서 청크가 추가되었습니다.")
    
    def query(self, question: str, thread_id: str = "default") -> dict:
        """질문에 대한 답변 생성"""
        initial_state = {
            "messages": [],
            "query": question,
            "context": "",
            "answer": "",
            "sources": []
        }
        
        # Graph 실행
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(initial_state, config)
        
        return result
    
    def clear_vectorstore(self):
        """벡터 스토어 초기화"""
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
            print("벡터 스토어가 초기화되었습니다.")
        
        # 새로운 벡터 스토어 생성
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
