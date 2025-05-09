```
# 뉴스레터 관리 시스템

이 시스템은 뉴스레터 구독자 관리와 뉴스레터 발송을 위한 웹 기반 애플리케이션입니다.

## 기능

- 관리자 로그인 시스템 (보안 강화)
- 구독자 관리 (추가, 조회, 수정, 삭제, 검색)
- OpenAI API를 활용한 뉴스 자동 크롤링 및 요약
- 중복 뉴스 필터링 및 유사도 검사
- API 토큰 사용량 및 비용 추적 기능
- Gmail API를 통한 이메일 발송 기능
- 모바일 대응 반응형 UI
- CSRF 보호 및 세션 관리

## 설치 방법

1. 레포지토리를 클론합니다:
```bash
git clone [레포지토리 URL]
cd news
```

2. 필요한 패키지를 설치합니다:
```bash
pip install -r requirements.txt
```

3. `.env` 파일을 설정합니다 (아래 환경 변수 섹션 참조)

4. 애플리케이션을 실행합니다:
```bash
python app.py
```

5. 뉴스 크롤링 및 발송을 실행합니다:
```bash
python news_crawler.py
```

## 파일 구조

- `app.py`: 웹 애플리케이션 메인 파일 (구독자 관리 CRUD)
- `news_crawler.py`: 뉴스 크롤링, 요약 및 이메일 발송 시스템
- `templates/`: 웹 UI 템플릿 파일
  - `base.html`: 기본 레이아웃 템플릿
  - `login.html`: 관리자 로그인 페이지
  - `list_subscribers.html`: 구독자 목록 페이지
  - `subscriber_form.html`: 구독자 추가/수정 폼
  - `error.html`: 오류 페이지
- `subscribers.json`: 구독자 정보 저장 파일
- `generate_hash.py`: 관리자 비밀번호 해시 생성 유틸리티
- `logs/`: 로그 파일 저장 디렉토리
- `requirements.txt`: 필요한 패키지 목록
- `.env`: 환경 변수 설정 파일

## 환경 변수 설정

`.env` 파일을 프로젝트 루트 디렉토리에 생성하고 다음 항목을 설정하세요:

```
# OpenAI API 설정
OPENAI_API_KEY=your_openai_api_key
NEWS_TOPICS=topic1,topic2,topic3
GPT_MODEL=gpt-4o-mini
USD_TO_KRW=1350

# 이메일 설정
EMAIL_USERNAME=your_email@gmail.com
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587

# 이메일 OAuth 2.0 인증 설정
GMAIL_CREDENTIALS_FILE=credentials.json
EMAIL_OAUTH2_TOKEN=token.json

# Flask 보안 설정
SECRET_KEY=your_secret_key_here
CSRF_SECRET_KEY=your_csrf_secret_key_here
ADMIN_USERNAME=your_admin_username
ADMIN_PASSWORD_HASH=your_bcrypt_hashed_password

# 서버 설정
DEBUG=True
HOST=0.0.0.0
PORT=5000
```

### 환경 변수 설명

- `OPENAI_API_KEY`: OpenAI API 키
- `NEWS_TOPICS`: 기본 뉴스 토픽 (쉼표로 구분)
- `GPT_MODEL`: 사용할 OpenAI 모델 (기본값: gpt-4o-mini)
- `USD_TO_KRW`: 달러 대 원화 환율 (기본값: 1350)
- `SECRET_KEY`: Flask 세션 암호화 키
- `CSRF_SECRET_KEY`: CSRF 보호를 위한 키
- `ADMIN_USERNAME`: 관리자 로그인 사용자 이름 (기본값: kca)
- `ADMIN_PASSWORD_HASH`: bcrypt로 해시된 관리자 비밀번호

## 주요 기능 설명

### 구독자 관리

- **추가**: 새로운 구독자 정보를 추가할 수 있습니다.
- **조회**: 등록된 모든 구독자 목록을 확인할 수 있습니다.
- **수정**: 구독자 정보(이름, 이메일, 관심 토픽)를 수정할 수 있습니다.
- **삭제**: 구독자 정보를 시스템에서 삭제할 수 있습니다.
- **검색**: 이름, 이메일, 관심 토픽으로 구독자를 검색할 수 있습니다.

### 뉴스 크롤링 및 요약

`news_crawler.py`는 다음과 같은 기능을 제공합니다:

1. **뉴스 수집**: 각 구독자의 관심 토픽에 맞는 뉴스를 수집합니다.
2. **본문 추출**: 뉴스 기사 본문을 추출합니다.
3. **요약 생성**: OpenAI API를 사용하여 뉴스 기사를 요약합니다.
4. **중복 제거**: 유사도 검사를 통해 중복 기사를 필터링합니다.
5. **이메일 발송**: 구독자별로 맞춤화된 뉴스레터를 발송합니다.

### API 토큰 사용량 추적

이 시스템은 OpenAI API 사용 시 토큰 사용량과 비용을 자동으로 추적합니다:

1. **실시간 추적**: 각 API 호출의 토큰 사용량과 비용을 실시간으로 추적합니다.
2. **통합 보고서**: 프로그램 종료 시 통합 보고서를 생성하여 로그 파일에 기록합니다.
3. **모델별 통계**: 사용한 모델별로 토큰 사용량과 비용을 세분화하여 제공합니다.
4. **비용 계산**: 달러 및 원화로 환산된 비용을 정확하게 계산합니다.

로그 파일은 `logs/` 디렉토리에 저장되며, 다음 형식으로 찾을 수 있습니다:
```
logs/news_crawler_[TIMESTAMP].log
logs/newsletter.log
```

## 보안 기능

이 애플리케이션은 다음과 같은 보안 기능을 포함하고 있습니다:

1. **비밀번호 해싱**: bcrypt를 사용하여 관리자 비밀번호를 안전하게 해싱하여 저장합니다.
2. **CSRF 보호**: Flask-WTF를 통해 Cross-Site Request Forgery 공격으로부터 보호합니다.
3. **세션 관리**: 세션 타임아웃(1시간) 및 안전한 세션 관리를 통해 인증 보안을 강화합니다.
4. **로깅**: 모든 중요 이벤트에 대한 로깅을 제공하여 보안 모니터링을 지원합니다.

## 해시된 비밀번호 생성 방법

새로운 관리자 비밀번호를 설정하려면 다음 명령어를 실행하세요:

```bash
python generate_hash.py
```

이 스크립트는 비밀번호를 입력받아 bcrypt로 해싱한 후 `.env` 파일에 저장할 수 있는 형식으로 출력합니다.

## 라이선스

이 프로젝트는 [사용 라이선스 입력] 라이선스를 따릅니다.
```