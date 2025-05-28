import fitz  # PyMuPDF
import re
from typing import List, Dict, Tuple
import os


class PDFToMarkdownConverter:
    """PDF 파일을 Markdown으로 변환하는 클래스"""
    
    def __init__(self):
        self.heading_sizes = {}  # 제목 크기 매핑을 저장
        
    def pdf_to_markdown(self, pdf_path: str) -> str:
        """
        PDF 파일을 읽어 Markdown 형식으로 변환
        
        Args:
            pdf_path (str): PDF 파일 경로
            
        Returns:
            str: Markdown 형식의 텍스트
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
            
        # PDF 문서 열기
        doc = fitz.open(pdf_path)
        markdown_content = []
        
        # 각 페이지 처리
        for page_num, page in enumerate(doc):
            page_content = self._process_page(page, page_num + 1)
            if page_content:
                markdown_content.append(page_content)
        
        doc.close()
        
        # 페이지 구분선으로 연결
        return "\n\n---\n\n".join(markdown_content)
    
    def _process_page(self, page, page_num: int) -> str:
        """
        단일 페이지를 처리하여 Markdown으로 변환
        
        Args:
            page: PyMuPDF 페이지 객체
            page_num (int): 페이지 번호
            
        Returns:
            str: 페이지의 Markdown 내용
        """
        # 텍스트 블록 추출
        blocks = page.get_text("dict")
        page_content = []
        
        # 페이지 번호 추가 (선택사항)
        # page_content.append(f"<!-- 페이지 {page_num} -->\n")
        
        # 블록별로 처리
        for block in blocks["blocks"]:
            if block["type"] == 0:  # 텍스트 블록
                block_text = self._process_text_block(block)
                if block_text:
                    page_content.append(block_text)
        
        return "\n\n".join(page_content)
    
    def _process_text_block(self, block: Dict) -> str:
        """
        텍스트 블록을 처리하여 Markdown 형식으로 변환
        
        Args:
            block (Dict): 텍스트 블록 정보
            
        Returns:
            str: Markdown 형식의 텍스트
        """
        lines = []
        
        for line in block.get("lines", []):
            line_text = ""
            line_size = 0
            line_flags = 0
            
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                    
                size = span.get("size", 0)
                flags = span.get("flags", 0)
                
                # 라인의 대표 크기와 플래그 설정
                if size > line_size:
                    line_size = size
                    line_flags = flags
                
                line_text += text + " "
            
            line_text = line_text.strip()
            if line_text:
                # 텍스트 스타일에 따라 Markdown 형식 적용
                formatted_text = self._apply_markdown_formatting(
                    line_text, line_size, line_flags
                )
                lines.append(formatted_text)
        
        return "\n".join(lines)
    
    def _apply_markdown_formatting(self, text: str, size: float, flags: int) -> str:
        """
        텍스트에 Markdown 형식 적용
        
        Args:
            text (str): 원본 텍스트
            size (float): 폰트 크기
            flags (int): 폰트 플래그 (굵게, 기울임 등)
            
        Returns:
            str: Markdown 형식이 적용된 텍스트
        """
        # 폰트 크기에 따른 제목 레벨 결정
        if size > 20:
            return f"# {text}"
        elif size > 16:
            return f"## {text}"
        elif size > 14:
            return f"### {text}"
        
        # 폰트 스타일 적용
        if flags & 2**4:  # 굵게
            text = f"**{text}**"
        if flags & 2**1:  # 기울임
            text = f"*{text}*"
        
        # 리스트 항목 감지
        if re.match(r'^[\•\-\*]\s', text):
            text = re.sub(r'^[\•\-\*]\s', '- ', text)
        elif re.match(r'^\d+\.\s', text):
            # 번호 매기기 리스트는 그대로 유지
            pass
        
        return text
    
    def _extract_images(self, page, output_dir: str = "images") -> List[str]:
        """
        페이지에서 이미지 추출 (선택적 기능)
        
        Args:
            page: PyMuPDF 페이지 객체
            output_dir (str): 이미지 저장 디렉토리
            
        Returns:
            List[str]: 추출된 이미지 파일 경로 목록
        """
        image_list = page.get_images()
        extracted_images = []
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            pix = fitz.Pixmap(page.parent, xref)
            
            if pix.n - pix.alpha < 4:  # GRAY 또는 RGB
                img_path = f"{output_dir}/image_p{page.number}_{img_index}.png"
                pix.save(img_path)
                extracted_images.append(img_path)
            else:  # CMYK 등 다른 색상 공간
                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                img_path = f"{output_dir}/image_p{page.number}_{img_index}.png"
                pix1.save(img_path)
                extracted_images.append(img_path)
                pix1 = None
            
            pix = None
        
        return extracted_images


def pdf_to_markdown(pdf_path: str) -> str:
    """
    PDF 파일을 Markdown으로 변환하는 간단한 함수
    
    Args:
        pdf_path (str): PDF 파일 경로
        
    Returns:
        str: Markdown 형식의 텍스트
    """
    converter = PDFToMarkdownConverter()
    return converter.pdf_to_markdown(pdf_path)


# 사용 예시
if __name__ == "__main__":
    # 테스트용 코드
    pdf_file = "example.pdf"  # 실제 PDF 파일 경로로 변경
    
    try:
        markdown_content = pdf_to_markdown(pdf_file)
        
        # 결과를 파일로 저장
        output_file = pdf_file.replace('.pdf', '.md')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
            
        print(f"변환 완료: {output_file}")
        print("\n--- 변환된 내용 미리보기 ---")
        print(markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content)
        
    except FileNotFoundError as e:
        print(f"오류: {e}")
    except Exception as e:
        print(f"변환 중 오류 발생: {e}")
