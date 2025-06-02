"""
RAG Graph 사용 예제
"""

from rag_graph import RAGGraph
import os

def main():
    # RAG Graph 인스턴스 생성
    rag = RAGGraph(persist_directory="./chroma_db")
    
    # 예제 1: 문서 추가
    print("=== 문서 추가 예제 ===")
    # 문서 파일 경로 리스트 (실제 파일 경로로 변경 필요)
    documents = [
        # "path/to/document1.pdf",
        # "path/to/document2.txt"
    ]
    
    if documents:
        rag.add_documents(documents)
    else:
        print("추가할 문서가 없습니다. 문서 경로를 지정해주세요.")
    
    # 예제 2: 질문하기
    print("\n=== 질문 예제 ===")
    questions = [
        "이 문서의 주요 내용은 무엇인가요?",
        "문서에서 언급된 핵심 개념들을 설명해주세요.",
        "저자가 제시한 주요 논점은 무엇인가요?"
    ]
    
    for question in questions:
        print(f"\n질문: {question}")
        result = rag.query(question)
        print(f"답변: {result['answer']}")
        print(f"출처: {', '.join(result['sources'])}")
    
    # 예제 3: 대화형 세션
    print("\n=== 대화형 세션 예제 ===")
    print("'quit'을 입력하면 종료됩니다.")
    
    thread_id = "conversation_1"  # 대화 세션 ID
    
    while True:
        user_input = input("\n질문을 입력하세요: ")
        if user_input.lower() == 'quit':
            break
        
        result = rag.query(user_input, thread_id=thread_id)
        print(f"\n답변: {result['answer']}")
        if result['sources']:
            print(f"출처: {', '.join(result['sources'])}")

if __name__ == "__main__":
    # OpenAI API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("경고: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print(".env 파일을 생성하고 API 키를 설정해주세요.")
        print("예: OPENAI_API_KEY=your-api-key-here")
    else:
        main()
