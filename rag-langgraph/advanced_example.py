"""
고급 RAG Graph 사용 예제
"""

from advanced_rag_graph import AdvancedRAGGraph
import os
from datetime import datetime

def main():
    # 고급 RAG Graph 인스턴스 생성
    rag = AdvancedRAGGraph(persist_directory="./advanced_chroma_db")
    
    print("=== 고급 RAG 시스템 예제 ===\n")
    
    # 예제 1: 디렉토리에서 문서 일괄 추가
    print("1. 디렉토리에서 문서 추가")
    # documents_dir = "./documents"  # 실제 문서가 있는 디렉토리로 변경
    # if os.path.exists(documents_dir):
    #     rag.add_documents_from_directory(documents_dir, glob_pattern="**/*.txt")
    # else:
    #     print(f"문서 디렉토리를 찾을 수 없습니다: {documents_dir}")
    
    # 예제 2: 다양한 검색 전략 테스트
    print("\n2. 다양한 검색 전략 테스트")
    test_queries = [
        {
            "query": "머신러닝의 기본 개념을 설명해주세요",
            "search_type": "similarity"
        },
        {
            "query": "딥러닝과 관련된 다양한 기술들을 알려주세요",
            "search_type": "mmr"  # 다양성 중시
        },
        {
            "query": "transformer 아키텍처의 구조와 특징",
            "search_type": "hybrid"  # 키워드 + 의미 검색
        }
    ]
    
    for test in test_queries:
        print(f"\n질문: {test['query']}")
        print(f"검색 전략: {test['search_type']}")
        
        result = rag.query_with_metadata(
            question=test['query'],
            search_type=test['search_type']
        )
        
        print(f"답변: {result['answer'][:200]}...")
        print(f"사용된 검색 전략: {result['search_type']}")
        if result['rerank_scores']:
            print(f"재순위 점수: {result['rerank_scores']}")
    
    # 예제 3: 메타데이터 필터링을 사용한 검색
    print("\n\n3. 메타데이터 필터링 검색")
    
    # 특정 파일 타입만 검색
    metadata_filter = {
        "file_type": ".txt"
    }
    
    result = rag.query_with_metadata(
        question="최신 기술 동향은 무엇인가요?",
        metadata_filter=metadata_filter,
        search_type="similarity"
    )
    
    print(f"질문: 최신 기술 동향은 무엇인가요?")
    print(f"필터: .txt 파일만 검색")
    print(f"답변: {result['answer'][:200]}...")
    
    # 예제 4: 대화형 세션 with 고급 기능
    print("\n\n4. 고급 대화형 세션")
    print("'quit'을 입력하면 종료됩니다.")
    print("'stats'를 입력하면 통계를 확인합니다.")
    print("검색 전략을 지정하려면 '[전략]:질문' 형식으로 입력하세요.")
    print("예: 'mmr:다양한 프로그래밍 언어들을 소개해주세요'")
    
    thread_id = f"advanced_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    while True:
        user_input = input("\n질문을 입력하세요: ")
        
        if user_input.lower() == 'quit':
            break
        
        if user_input.lower() == 'stats':
            stats = rag.get_statistics()
            print("\n=== 벡터 스토어 통계 ===")
            for key, value in stats.items():
                print(f"{key}: {value}")
            continue
        
        # 검색 전략 파싱
        search_type = "similarity"
        question = user_input
        
        if ':' in user_input:
            parts = user_input.split(':', 1)
            if parts[0].lower() in ['similarity', 'mmr', 'hybrid']:
                search_type = parts[0].lower()
                question = parts[1].strip()
        
        # 쿼리 실행
        result = rag.query_with_metadata(
            question=question,
            search_type=search_type,
            thread_id=thread_id
        )
        
        print(f"\n답변: {result['answer']}")
        if result['sources']:
            print(f"출처: {', '.join(result['sources'])}")
        print(f"검색 전략: {result['search_type']}")
        if result['rerank_scores']:
            print(f"문서 관련성 점수: {[f'{score:.1f}' for score in result['rerank_scores']]}")

def create_sample_documents():
    """테스트용 샘플 문서 생성"""
    os.makedirs("./sample_documents", exist_ok=True)
    
    documents = [
        {
            "filename": "ml_basics.txt",
            "content": """머신러닝 기초

머신러닝은 컴퓨터가 명시적으로 프로그래밍되지 않고도 학습할 수 있는 능력을 부여하는 인공지능의 한 분야입니다.

주요 유형:
1. 지도 학습 (Supervised Learning): 레이블이 있는 데이터로 학습
2. 비지도 학습 (Unsupervised Learning): 레이블이 없는 데이터로 패턴 발견
3. 강화 학습 (Reinforcement Learning): 보상을 통한 학습

핵심 개념:
- 특징 (Features): 입력 데이터의 속성
- 레이블 (Labels): 예측하고자 하는 목표값
- 모델 (Model): 데이터로부터 학습한 패턴
- 훈련 (Training): 모델이 데이터로부터 학습하는 과정
"""
        },
        {
            "filename": "deep_learning.txt",
            "content": """딥러닝 개요

딥러닝은 인공 신경망을 기반으로 하는 머신러닝의 한 분야입니다. 여러 층의 뉴런으로 구성된 네트워크를 사용합니다.

주요 아키텍처:
1. CNN (Convolutional Neural Networks): 이미지 처리에 특화
2. RNN (Recurrent Neural Networks): 순차 데이터 처리
3. Transformer: 자연어 처리의 혁명
4. GAN (Generative Adversarial Networks): 생성 모델

응용 분야:
- 컴퓨터 비전
- 자연어 처리
- 음성 인식
- 추천 시스템
"""
        },
        {
            "filename": "transformer_architecture.txt",
            "content": """Transformer 아키텍처

Transformer는 2017년 "Attention is All You Need" 논문에서 소개된 혁신적인 신경망 아키텍처입니다.

핵심 구성 요소:
1. Self-Attention 메커니즘: 입력 시퀀스의 모든 위치를 동시에 고려
2. Multi-Head Attention: 여러 관점에서 정보를 병렬 처리
3. Positional Encoding: 순서 정보 인코딩
4. Feed-Forward Networks: 각 위치에서 독립적으로 적용

장점:
- 병렬 처리 가능
- 장거리 의존성 포착
- 학습 속도 향상

대표 모델:
- BERT: 양방향 인코더
- GPT: 자동회귀 디코더
- T5: 인코더-디코더 구조
"""
        }
    ]
    
    for doc in documents:
        with open(f"./sample_documents/{doc['filename']}", 'w', encoding='utf-8') as f:
            f.write(doc['content'])
    
    print(f"샘플 문서 {len(documents)}개가 생성되었습니다.")
    return "./sample_documents"

if __name__ == "__main__":
    # OpenAI API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("경고: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print(".env 파일을 생성하고 API 키를 설정해주세요.")
        print("\n샘플 문서를 생성하려면 'y'를 입력하세요:")
        
        if input().lower() == 'y':
            sample_dir = create_sample_documents()
            print(f"\n샘플 문서가 {sample_dir}에 생성되었습니다.")
            print("API 키를 설정한 후 다시 실행해주세요.")
    else:
        main()
