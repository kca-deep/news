# 뉴스 크롤러 (News Crawler)

최근 2일 이내의 Google News 기사를 수집하고 AI로 요약하는 프로그램입니다.

## 주요 기능

- Google News RSS 피드에서 최신 뉴스 수집
- 여러 검색어(토픽)에 대한 병렬 처리 지원
- OpenAI GPT-4o-mini를 이용한 뉴스 기사 요약
- 진행 상황 실시간 표시
- 결과를 텍스트 및 JSON 형식으로 저장
- 검색 결과가 없는 토픽 정보 제공
- API 토큰 사용량 및 비용 추적

## 설치 방법

1. Python 3.13.2 이상 설치
2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 환경 설정

1. `.env` 파일 생성 후 아래 내용 추가:
```
OPENAI_API_KEY=your_openai_api_key_here
NEWS_TOPICS=토픽1,토픽2,토픽3
```

- `OPENAI_API_KEY`: OpenAI API 키
- `NEWS_TOPICS`: 검색할 뉴스 토픽 (쉼표로 구분)

## 사용 방법

```bash
python news_crawler.py
```

## 출력 결과

1. 실시간 진행 상황
```
전체 진행률: 67%|██████████████████████| 2/3 [00:45<00:22, 22.50초/주제]
```

2. 결과 파일 (`result` 디렉토리)
- `result_YYYYMMDD_HHMMSS.txt`: 텍스트 형식 결과
- `result_YYYYMMDD_HHMMSS.json`: JSON 형식 결과

### 결과 파일 포함 정보
- 뉴스 제목, 출처, 게시일, 링크
- AI 요약 내용
- 검색 결과가 없는 토픽 목록
- OpenAI API 토큰 사용량 및 비용 정보

## 주요 설정

- 검색 기간: 최근 2일
- 토픽당 최대 기사 수: 3개
- 병렬 처리 작업자 수: 3
- GPT 모델: gpt-4o-mini
- 기사 본문 최대 길이: 2,000자

## 의존성 패키지

```
requests==2.31.0
python-dotenv==1.0.0
pathlib==1.0.1
email-validator==2.1.0
beautifulsoup4==4.12.2
concurrent-log-handler==0.9.24
tqdm==4.66.1
```

## 기능

- Google News RSS 피드에서 검색어 기반 최신 뉴스 수집
- 최근 2일 이내의 뉴스만 필터링
- 국가 및 언어별 뉴스 필터링
- OpenAI API를 사용한 뉴스 기사 요약
- 수집된 뉴스 정보 콘솔 출력
- 결과를 텍스트 파일로 저장 (result 폴더 내 타임스탬프 포함 파일명)
- OpenAI API 토큰 사용량 및 비용 추적 (달러 및 원화로 표시)

## Windows PC에서 VSCode 없이 실행하기

### 1. Python 설치

1. [Python 공식 웹사이트](https://www.python.org/downloads/windows/)에서 최신 버전의 Python을 다운로드합니다.
2. 설치 시 "Add Python to PATH" 옵션을 체크하세요.
3. 설치가 완료되면 명령 프롬프트(cmd)를 열고 다음 명령어로 Python이 제대로 설치되었는지 확인합니다:
   ```
   python --version
   ```

### 2. 프로젝트 파일 준비

1. 프로젝트 파일을 담을 폴더를 생성합니다 (예: `C:\news_crawler`).
2. 다음 파일들을 해당 폴더에 복사합니다:
   - `news_crawler.py`
   - `requirements.txt`
   - `.gitignore` (선택사항)

3. 같은 폴더에 `.env` 파일을 생성하고 다음 내용을 입력합니다:
   ```
   # 뉴스 검색어 설정 (쉼표로 구분)
   NEWS_TOPICS=ICT기금,인공지능,빅데이터
   
   # OpenAI API 키 설정
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   **중요**: `your_openai_api_key_here` 부분을 실제 OpenAI API 키로 교체하세요.

### 3. 가상 환경 설정 및 라이브러리 설치

1. 명령 프롬프트(cmd)를 열고 프로젝트 폴더로 이동합니다:
   ```
   cd C:\news_crawler
   ```

2. 가상 환경을 생성합니다:
   ```
   python -m venv venv
   ```

3. 가상 환경을 활성화합니다:
   ```
   venv\Scripts\activate
   ```
   명령 프롬프트 앞에 `(venv)`가 표시되면 가상 환경이 활성화된 것입니다.

4. 필요한 라이브러리를 설치합니다:
   ```
   pip install -r requirements.txt
   ```

### 4. 프로그램 실행

1. 가상 환경이 활성화된 상태에서 다음 명령어로 프로그램을 실행합니다:
   ```
   python news_crawler.py
   ```

2. 프로그램이 실행되면 각 검색어에 대한 최근 2일 이내의 뉴스를 가져와 처리하고, 결과를 `result` 폴더에 저장합니다.

### 5. 실행 스크립트 생성 (선택사항)

매번 명령어를 입력하지 않고 더 쉽게 실행하려면 배치 파일(`.bat`)을 생성할 수 있습니다:

1. 메모장을 열고 다음 내용을 입력합니다:
   ```batch
   @echo off
   cd /d %~dp0
   call venv\Scripts\activate
   python news_crawler.py
   pause
   ```

2. 이 파일을 `run_news_crawler.bat`라는 이름으로 프로젝트 폴더에 저장합니다.

3. 이제 이 배치 파일을 더블 클릭하면 프로그램이 자동으로 실행됩니다.

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

- `display_news(news_items, limit=5)`: 뉴스 기사 목록을 콘솔에 출력하는 함수
  - `news_items`: 뉴스 기사 목록
  - `limit`: 출력할 기사 수 (기본값: 5)

- `save_to_file(news_items, result_dir="result", include_token_info=True)`: 뉴스 기사 목록을 파일에 저장하는 함수
  - `news_items`: 뉴스 기사 목록
  - `result_dir`: 결과 파일을 저장할 디렉토리 (기본값: "result")
  - `include_token_info`: 토큰 사용량 정보 포함 여부

- `calculate_token_cost()`: OpenAI API 토큰 사용량 및 비용을 계산하는 함수

- `display_token_info()`: OpenAI API 토큰 사용량 및 비용 정보를 콘솔에 출력하는 함수

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

### 토큰 사용량 및 비용 정보

프로그램 실행이 완료되면 OpenAI API 토큰 사용량 및 비용 정보가 콘솔과 결과 파일에 표시됩니다:

```
================================================================================
                        OpenAI API 토큰 사용량 정보                        
================================================================================
프롬프트 토큰: 1,234개
완성 토큰: 567개
총 토큰: 1,801개

예상 비용: $0.003851 (약 5.20원)
참고: https://platform.openai.com/settings/organization/limits
================================================================================
```

이 정보를 통해 API 사용량을 모니터링하고 비용을 관리할 수 있습니다.

## Git 관리

이 프로젝트는 다음 파일/폴더를 Git에서 제외합니다:

- `.env` 파일: 개인 API 키와 같은 민감한 정보가 포함되어 있습니다.
- `result/` 폴더: 프로그램 실행 결과가 저장되는 폴더입니다.

이러한 설정은 `.gitignore` 파일에 정의되어 있습니다.

## 문제 해결

1. **Python이 인식되지 않는 경우**:
   - Python이 PATH에 추가되었는지 확인하세요.
   - 또는 전체 경로를 사용하여 Python을 실행하세요 (예: `C:\Python39\python.exe`).

2. **라이브러리 설치 오류**:
   - pip를 최신 버전으로 업데이트해보세요: `python -m pip install --upgrade pip`
   - 인터넷 연결을 확인하세요.

3. **OpenAI API 오류**:
   - `.env` 파일에 올바른 API 키가 설정되어 있는지 확인하세요.
   - API 키가 유효한지 확인하세요.

4. **결과 폴더가 생성되지 않는 경우**:
   - 수동으로 `result` 폴더를 생성해보세요.
   - 프로그램을 실행하는 사용자에게 폴더 생성 권한이 있는지 확인하세요.