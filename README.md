# Google News RSS 크롤러

이 프로젝트는 Google News RSS 피드를 활용하여 특정 검색어에 대한 최신 뉴스를 수집하고, OpenAI API를 사용하여 뉴스 기사를 요약하는 Python 스크립트입니다.

## 기능

- Google News RSS 피드에서 검색어 기반 최신 뉴스 수집
- 최근 2일 이내의 뉴스만 필터링
- 국가 및 언어별 뉴스 필터링
- OpenAI API를 사용한 뉴스 기사 요약
- 수집된 뉴스 정보 콘솔 출력
- 결과를 텍스트 파일로 저장 (result 폴더 내 타임스탬프 포함 파일명)

## 설치 방법

1. 필요한 라이브러리 설치:
```bash
pip install -r requirements.txt
```

2. `.env` 파일 설정:
```
# 뉴스 검색어 설정 (쉼표로 구분)
NEWS_TOPICS=ICT기금,인공지능,빅데이터

# OpenAI API 키 설정
OPENAI_API_KEY=your_openai_api_key_here
```

## 사용 방법

기본 사용법:
```bash
python news_crawler.py
```

## 주요 함수

- `get_google_news(query='', country='kr', language='ko')`: Google News RSS 피드에서 검색어 기반으로 뉴스를 가져오는 함수
  - `query`: 검색어 (예: 'ICT기금', '인공지능' 등)
  - `country`: 국가 코드 (예: 'kr', 'us', 'jp' 등)
  - `language`: 언어 코드 (예: 'ko', 'en', 'ja' 등)

- `get_article_content(url)`: 뉴스 기사 URL에서 본문 내용을 가져오는 함수
  - `url`: 뉴스 기사 URL

- `summarize_with_openai(text, title)`: OpenAI API를 사용하여 텍스트를 요약하는 함수
  - `text`: 요약할 텍스트
  - `title`: 기사 제목

- `display_news(news_items, limit=10)`: 뉴스 기사 목록을 콘솔에 출력하는 함수
  - `news_items`: 뉴스 기사 목록
  - `limit`: 출력할 기사 수 (기본값: 10)

- `save_to_file(news_items, result_dir="result")`: 뉴스 기사 목록을 파일에 저장하는 함수
  - `news_items`: 뉴스 기사 목록
  - `result_dir`: 결과 파일을 저장할 디렉토리 (기본값: "result")

## 기술 스택

- Python 3.x
- `requests`: HTTP 요청 처리
- `xml.etree.ElementTree`: XML 파싱 (RSS 피드 처리)
- `python-dotenv`: 환경 변수 관리
- `pathlib`: 파일 경로 처리
- 정규 표현식 (`re` 모듈): 텍스트 처리
- OpenAI API: 텍스트 요약

## 출력 결과

프로그램 실행 결과는 `result` 폴더 내에 `result_yyyymmdd_HHMMSS.txt` 형식의 파일명으로 저장됩니다. 각 뉴스 기사에 대한 정보와 AI 요약이 포함됩니다.

## Git 관리

이 프로젝트는 다음 파일/폴더를 Git에서 제외합니다:

- `.env` 파일: 개인 API 키와 같은 민감한 정보가 포함되어 있습니다.
- `result/` 폴더: 프로그램 실행 결과가 저장되는 폴더입니다.

이러한 설정은 `.gitignore` 파일에 정의되어 있습니다.