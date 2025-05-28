# PDF to Markdown Converter

PDF 파일을 Markdown 형식으로 변환하는 Python 스크립트입니다. PyMuPDF 라이브러리를 사용하여 PDF의 텍스트를 추출하고, 폰트 크기와 스타일을 분석하여 적절한 Markdown 형식으로 변환합니다.

## 주요 기능

- 📄 PDF 파일을 Markdown(.md) 파일로 변환
- 📝 폰트 크기에 따른 제목 레벨 자동 설정 (H1, H2, H3)
- ✨ 텍스트 스타일 유지 (굵게, 기울임)
- 📋 리스트 항목 자동 감지 및 변환
- 🖼️ 이미지 추출 기능 (선택적)
- 📑 페이지별 구분 처리

## 설치 방법

### 1. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

또는 직접 설치:

```bash
pip install PyMuPDF==1.23.8
```

### 2. 스크립트 다운로드

`pdf2markdown.py` 파일을 프로젝트 디렉토리에 다운로드합니다.

## 사용 방법

### 기본 사용법

#### 1. 명령줄에서 직접 실행

```python
python pdf2markdown.py
```

스크립트 내의 `example.pdf`를 실제 PDF 파일 경로로 변경한 후 실행합니다.

#### 2. Python 코드에서 사용

```python
from pdf2markdown import pdf_to_markdown

# PDF 파일을 Markdown으로 변환
pdf_path = "your_document.pdf"
markdown_content = pdf_to_markdown(pdf_path)

# 결과를 파일로 저장
output_path = "output.md"
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(markdown_content)

print(f"변환 완료: {output_path}")
```

### 고급 사용법

#### PDFToMarkdownConverter 클래스 사용

```python
from pdf2markdown import PDFToMarkdownConverter

# 변환기 인스턴스 생성
converter = PDFToMarkdownConverter()

# PDF 변환
pdf_path = "document.pdf"
markdown_content = converter.pdf_to_markdown(pdf_path)

# 결과 저장
with open("document.md", 'w', encoding='utf-8') as f:
    f.write(markdown_content)
```

#### 이미지 추출 기능 사용

```python
import fitz
from pdf2markdown import PDFToMarkdownConverter

converter = PDFToMarkdownConverter()
doc = fitz.open("document.pdf")

# 특정 페이지에서 이미지 추출
page = doc[0]  # 첫 번째 페이지
images = converter._extract_images(page, output_dir="extracted_images")

print(f"추출된 이미지: {images}")
doc.close()
```

## 클래스 및 메서드 설명

### PDFToMarkdownConverter 클래스

PDF를 Markdown으로 변환하는 메인 클래스입니다.

#### 주요 메서드

- **`pdf_to_markdown(pdf_path: str) -> str`**
  - PDF 파일을 읽어 Markdown 형식으로 변환
  - 매개변수: `pdf_path` - PDF 파일 경로
  - 반환값: Markdown 형식의 텍스트

- **`_process_page(page, page_num: int) -> str`**
  - 단일 페이지를 처리하여 Markdown으로 변환
  - 내부 메서드

- **`_process_text_block(block: Dict) -> str`**
  - 텍스트 블록을 처리하여 Markdown 형식으로 변환
  - 내부 메서드

- **`_apply_markdown_formatting(text: str, size: float, flags: int) -> str`**
  - 텍스트에 Markdown 형식 적용
  - 폰트 크기와 스타일에 따라 적절한 형식 결정

- **`_extract_images(page, output_dir: str = "images") -> List[str]`**
  - 페이지에서 이미지 추출 (선택적 기능)
  - 매개변수: `output_dir` - 이미지 저장 디렉토리
  - 반환값: 추출된 이미지 파일 경로 목록

### 간편 함수

- **`pdf_to_markdown(pdf_path: str) -> str`**
  - PDF 파일을 Markdown으로 변환하는 간단한 함수
  - 내부적으로 PDFToMarkdownConverter 클래스 사용

## 변환 규칙

### 제목 레벨
- 폰트 크기 > 20pt: `# H1 제목`
- 폰트 크기 > 16pt: `## H2 제목`
- 폰트 크기 > 14pt: `### H3 제목`

### 텍스트 스타일
- 굵은 텍스트: `**굵은 텍스트**`
- 기울임 텍스트: `*기울임 텍스트*`

### 리스트
- 불릿 리스트 (•, -, *): `- 항목`
- 번호 리스트: `1. 항목` (원본 유지)

### 페이지 구분
- 각 페이지는 `---`로 구분됩니다

## 예제

### 기본 변환 예제

```python
from pdf2markdown import pdf_to_markdown

# PDF 변환
markdown = pdf_to_markdown("report.pdf")

# 결과 저장
with open("report.md", 'w', encoding='utf-8') as f:
    f.write(markdown)
```

### 여러 PDF 파일 일괄 변환

```python
import os
from pdf2markdown import pdf_to_markdown

# PDF 파일이 있는 디렉토리
pdf_directory = "./pdfs"
output_directory = "./markdown"

# 출력 디렉토리 생성
os.makedirs(output_directory, exist_ok=True)

# 모든 PDF 파일 변환
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        md_filename = filename.replace(".pdf", ".md")
        md_path = os.path.join(output_directory, md_filename)
        
        try:
            markdown_content = pdf_to_markdown(pdf_path)
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"✅ 변환 완료: {filename} → {md_filename}")
        except Exception as e:
            print(f"❌ 변환 실패: {filename} - {e}")
```

## 주의사항

1. **복잡한 레이아웃**: 다단 구성, 표, 복잡한 레이아웃의 PDF는 완벽하게 변환되지 않을 수 있습니다.

2. **이미지 처리**: 이미지는 별도로 추출되며, Markdown 내에 자동으로 삽입되지 않습니다.

3. **표(Table) 변환**: 현재 버전에서는 표를 Markdown 표 형식으로 변환하지 않습니다.

4. **인코딩**: UTF-8 인코딩을 사용합니다. 특수 문자가 포함된 PDF의 경우 주의가 필요합니다.

5. **메모리 사용**: 대용량 PDF 파일의 경우 메모리 사용량이 높을 수 있습니다.

## 문제 해결

### FileNotFoundError
```
오류: PDF 파일을 찾을 수 없습니다: example.pdf
```
→ PDF 파일 경로가 올바른지 확인하세요.

### PyMuPDF 설치 오류
```bash
# Windows에서 설치 실패 시
pip install --upgrade pip
pip install PyMuPDF --no-cache-dir
```

### 한글 깨짐 문제
파일 저장 시 `encoding='utf-8'`을 명시적으로 지정하세요:
```python
with open("output.md", 'w', encoding='utf-8') as f:
    f.write(markdown_content)
```

## 라이선스

이 프로젝트는 PyMuPDF 라이브러리를 사용합니다. PyMuPDF의 라이선스 조건을 확인하세요.

## 기여하기

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다!

## 버전 정보

- 현재 버전: 1.0.0
- PyMuPDF 버전: 1.23.8
- Python 버전: 3.6 이상

---

📧 문의사항이나 제안사항이 있으시면 이슈를 등록해주세요.
