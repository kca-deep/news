# 뉴스레터 관리 시스템

이 시스템은 뉴스레터 구독자 관리와 뉴스레터 발송을 위한 웹 기반 애플리케이션입니다.

## 기능

- 관리자 로그인 시스템 (보안 강화)
- 구독자 관리 (추가, 조회, 삭제)
- OpenAI API를 활용한 뉴스 자동 요약
- API 토큰 사용량 및 비용 추적 기능
- 현대적인 UI (Shadcn UI + Tailwind CSS)
- 다크모드 지원

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

## 환경 변수 설정

`.env` 파일을 프로젝트 루트 디렉토리에 생성하고 다음 항목을 설정하세요:

```
OPENAI_API_KEY=your_openai_api_key
NEWS_TOPICS=topic1,topic2,topic3

# OpenAI API 설정
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
FLASK_SECRET_KEY=your_secret_key_here
ADMIN_PASSWORD_HASH=your_bcrypt_hashed_password
ADMIN_USERNAME=your_admin_username
```

### 환경 변수 설명

- `GPT_MODEL`: 사용할 OpenAI 모델 (기본값: gpt-4o-mini)
- `USD_TO_KRW`: 달러 대 원화 환율 (기본값: 1350)

## API 토큰 사용량 추적

이 시스템은 OpenAI API 사용 시 토큰 사용량과 비용을 자동으로 추적합니다:

1. **실시간 추적**: 각 API 호출의 토큰 사용량과 비용을 실시간으로 추적합니다.
2. **통합 보고서**: 프로그램 종료 시 통합 보고서를 생성하여 로그 파일에 기록합니다.
3. **모델별 통계**: 사용한 모델별로 토큰 사용량과 비용을 세분화하여 제공합니다.
4. **비용 계산**: 달러 및 원화로 환산된 비용을 정확하게 계산합니다.

로그 파일은 `logs/` 디렉토리에 저장되며, 다음 형식으로 찾을 수 있습니다:
```
logs/news_crawler_[TIMESTAMP].log
```

## 보안 기능

이 애플리케이션은 다음과 같은 보안 기능을 포함하고 있습니다:

1. **비밀번호 해싱**: bcrypt를 사용하여 관리자 비밀번호를 안전하게 해싱하여 저장합니다.
2. **CSRF 보호**: Flask-WTF를 통해 Cross-Site Request Forgery 공격으로부터 보호합니다.
3. **세션 관리**: 세션 타임아웃 및 안전한 세션 관리를 통해 인증 보안을 강화합니다.
4. **로깅**: 모든 중요 이벤트에 대한 로깅을 제공하여 보안 모니터링을 지원합니다.

## 해시된 비밀번호 생성 방법

새로운 관리자 비밀번호를 설정하려면 다음 명령어를 실행하세요:

```bash
python generate_hash.py
```

이 스크립트는 비밀번호를 입력받아 bcrypt로 해싱한 후 `.env` 파일에 저장할 수 있는 형식으로 출력합니다.

## 라이선스

이 프로젝트는 [사용 라이선스 입력] 라이선스를 따릅니다.