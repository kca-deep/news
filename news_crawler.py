#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import time
import json
import base64
import logging
import pathlib
import urllib.parse
import requests
import email.utils
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from tqdm import tqdm
import colorama

# Google API 관련 모듈
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# 환경 변수 및 로깅 설정
load_dotenv()
LOG_DIR = pathlib.Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"news_crawler_{TIMESTAMP}.log"

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8", errors="replace")
    ],
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("news_crawler")

# 터미널 색상 및 tqdm 진행바 설정
colorama.init(autoreset=True)
COLORS = {
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "MAGENTA": "\033[95m",
    "CYAN": "\033[96m",
    "WHITE": "\033[97m",
    "RESET": "\033[0m",
}


def colored_bar_format(color: str) -> str:
    return f"{color}{{desc}}: {{percentage:3.0f}}%|{{bar:30}}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]{COLORS['RESET']}"


def get_tqdm_settings(
    desc: str, color: str, position: int, leave: bool = False
) -> dict:
    return {
        "desc": desc,
        "ncols": 0,
        "bar_format": colored_bar_format(color),
        "dynamic_ncols": True,
        "mininterval": 0.3,
        "position": position,
        "leave": leave,
    }


# =============================================================================
# NewsFetcher 클래스: 뉴스 검색, 기사 본문 추출, 요약, 유사도 및 중복 제거 기능
# =============================================================================
class NewsFetcher:
    def __init__(self):
        self.news_cache: dict = {}
        self.similarity_cache: dict = {}
        self.api_usage_stats: dict = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost_usd": 0.0,
            "total_cost_krw": 0.0,
            "api_calls": 0,
            "models": {},
        }

    @staticmethod
    def preprocess_title(title: str) -> str:
        """뉴스 제목 전처리: 불필요한 접미사와 특수문자 제거 후 소문자화"""
        title = re.sub(r"\s*[-–]\s*[\w\s]+$", "", title)
        title = re.sub(r"[^\w\s]", " ", title)
        title = re.sub(r"\s+", " ", title)
        return title.strip().lower()

    @staticmethod
    def simple_jaccard_similarity(text1: str, text2: str) -> float:
        """두 텍스트 간의 Jaccard 유사도를 계산"""
        prep1 = NewsFetcher.preprocess_title(text1)
        prep2 = NewsFetcher.preprocess_title(text2)
        tokens1, tokens2 = set(prep1.split()), set(prep2.split())
        if not tokens1 or not tokens2:
            return 0.0
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        weight = (
            1.2 if min(len(tokens1), len(tokens2)) <= 3 and intersection > 0 else 1.0
        )
        similarity = (intersection / union) * weight
        return min(similarity, 1.0)

    def track_api_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        usd_cost: float,
        krw_cost: float,
    ) -> None:
        """API 사용량을 누적 기록"""
        stats = self.api_usage_stats
        stats["total_tokens"] += total_tokens
        stats["prompt_tokens"] += prompt_tokens
        stats["completion_tokens"] += completion_tokens
        stats["total_cost_usd"] += usd_cost
        stats["total_cost_krw"] += krw_cost
        stats["api_calls"] += 1

        if model not in stats["models"]:
            stats["models"][model] = {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_cost_usd": 0.0,
                "total_cost_krw": 0.0,
                "api_calls": 0,
            }
        model_stats = stats["models"][model]
        model_stats["total_tokens"] += total_tokens
        model_stats["prompt_tokens"] += prompt_tokens
        model_stats["completion_tokens"] += completion_tokens
        model_stats["total_cost_usd"] += usd_cost
        model_stats["total_cost_krw"] += krw_cost
        model_stats["api_calls"] += 1

    def get_google_news(
        self,
        query: str = "",
        country: str = "kr",
        language: str = "ko",
        max_items: int = 30,
    ) -> list:
        """
        Google News RSS를 사용해 최신 뉴스 기사를 검색 (최대 max_items개)
        """
        cache_key = f"{query}_{country}_{language}"
        if cache_key in self.news_cache:
            logger.info(f"캐시에서 토픽 '{query}' 뉴스 로드")
            return self.news_cache[cache_key]

        base_url = "https://news.google.com/rss"
        url = f"{base_url}/search?q={urllib.parse.quote(query)}" if query else base_url
        url += f"&hl={language}-{country}&gl={country}&ceid={country}:{language}"
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                logger.error(f"HTTP 상태 코드 {response.status_code}")
                return []
            root = ET.fromstring(response.content)
            news_items = []
            two_days_ago = datetime.now(timezone.utc) - timedelta(days=2)
            for item in root.findall(".//item"):
                title_elem = item.find("title")
                link_elem = item.find("link")
                pub_date_elem = item.find("pubDate")
                desc_elem = item.find("description")
                title = title_elem.text if title_elem is not None else "No Title"
                link = link_elem.text if link_elem is not None else "#"
                pub_date_text = (
                    pub_date_elem.text if pub_date_elem is not None else None
                )
                formatted_date = "No date info"
                is_recent = True
                parsed_date = None
                if pub_date_text:
                    try:
                        parsed_date = email.utils.parsedate_to_datetime(pub_date_text)
                        formatted_date = parsed_date.strftime("%Y-%m-%d %H:%M")
                        is_recent = parsed_date >= two_days_ago
                    except Exception:
                        logger.warning("날짜 파싱 오류")
                        formatted_date = pub_date_text
                if not is_recent:
                    continue
                source = None
                m = re.search(r"^(.*?)\s*-\s*([^-]+)$", title)
                if m:
                    source = m.group(2).strip()
                if not source and desc_elem is not None and desc_elem.text:
                    m = re.search(r'<font size="-1">([^<]+)</font>', desc_elem.text)
                    if m:
                        source = m.group(1).strip()
                    else:
                        m = re.search(r"<b>([^<]+)</b>", desc_elem.text)
                        if m:
                            source = m.group(1).strip()
                if not source:
                    source_elem = item.find("source")
                    if source_elem is not None:
                        source = source_elem.text.strip()
                news_items.append(
                    {
                        "title": title,
                        "link": link,
                        "published": formatted_date,
                        "published_raw": pub_date_text,
                        "published_datetime": parsed_date,
                        "source": source if source else "Unknown",
                        "query": query,
                        "summary": (
                            re.sub(r"<[^>]+>", "", desc_elem.text).strip()
                            if desc_elem is not None
                            else ""
                        ),
                    }
                )
                if len(news_items) >= max_items:
                    break
            self.news_cache[cache_key] = news_items
            return news_items
        except Exception as e:
            logger.error(f"뉴스 가져오기 오류: {e}")
            return []

    def get_article_content(self, url: str) -> tuple:
        """
        주어진 URL에서 기사 본문과 (있다면) 출처 정보를 추출
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                logger.error(f"HTTP 상태 코드 {response.status_code}")
                return "기사 내용을 가져올 수 없습니다.", None

            soup = BeautifulSoup(response.content, "html.parser")
            source_meta = soup.find("meta", property="og:site_name")
            source = source_meta.get("content") if source_meta else None

            content = ""
            article = soup.find("article")
            if article:
                content = article.get_text(strip=True)
            if not content:
                meta_desc = soup.find("meta", {"name": "description"}) or soup.find(
                    "meta", property="og:description"
                )
                if meta_desc:
                    content = meta_desc.get("content", "")
            if not content:
                content = " ".join(
                    tag.get_text(strip=True)
                    for tag in soup.find_all(
                        ["p", "div"],
                        class_=re.compile(r"article|content|body|text|main", re.I),
                    )
                    if tag.get_text(strip=True)
                )
            if content:
                content = re.sub(r"\s+", " ", content).strip()
                content = re.sub(
                    r"(광고|\[광고\]|sponsored content|AD).*?(?=\s|$)",
                    "",
                    content,
                    flags=re.I,
                )
                return (
                    content[:2000] + "..." if len(content) > 2000 else content
                ), source
            return "기사 내용을 추출할 수 없습니다.", source
        except Exception as e:
            logger.error(f"기사 내용 가져오기 오류: {e}")
            return "기사 내용을 가져오는 중 오류가 발생했습니다.", None

    def summarize_with_openai(self, title: str, text: str) -> str:
        """
        OpenAI API를 사용하여 기사 제목과 본문으로 300자 요약문을 생성
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API 키가 설정되지 않았습니다.")
            return "요약을 생성할 수 없습니다. API 키를 확인하세요."
        system_prompt = (
            "당신은 최고의 뉴스 에디터로, 뉴스 기사를 명확하고 풍부하게 요약하는 전문가입니다.\n\n"
            "뉴스 기사의 핵심 내용을 파악하여 통합된 하나의 요약문을 작성해주세요. 이 요약문은 다음 내용을 포함해야 합니다:\n\n"
            "- 기사의 핵심 주제 및 중요한 정보\n"
            "- 주요 사실, 관련 데이터, 중요한 인용구\n"
            "- 뉴스의 맥락적 중요성과 잠재적 영향\n\n"
            "작성 시 다음 원칙을 따르세요:\n"
            "- 정확성, 간결성, 객관성, 가독성, 완전성을 고려\n"
            "문단 구분은 자연스럽게 하되, 번호나 글머리 기호 없이 하나의 흐름으로 작성하고, 전체 요약은 정확히 300자로 작성하세요. 줄임표시(...)는 사용하지 마세요."
        )
        data = {
            "model": os.getenv("GPT_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"제목: {title}\n\n내용: {text}\n\n위 뉴스 기사를 요약해주세요.",
                },
            ],
            "temperature": 0.5,
            "max_tokens": 500,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.1,
        }
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json=data,
                timeout=30,
            )
            if response.status_code == 200:
                result = response.json()
                summary = result["choices"][0]["message"]["content"].strip()
                if "usage" in result:
                    prompt_tokens = result["usage"].get("prompt_tokens", 0)
                    completion_tokens = result["usage"].get("completion_tokens", 0)
                    total_tokens = result["usage"].get("total_tokens", 0)
                    model_prices = {
                        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
                        "gpt-4o": {"input": 5.0, "output": 15.0},
                        "gpt-4": {"input": 10.0, "output": 30.0},
                        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
                    }
                    price_info = model_prices.get(
                        data["model"], {"input": 0.15, "output": 0.60}
                    )
                    input_cost = (prompt_tokens / 1_000_000) * price_info["input"]
                    output_cost = (completion_tokens / 1_000_000) * price_info["output"]
                    total_cost = input_cost + output_cost
                    exchange_rate = float(os.getenv("USD_TO_KRW", 1350))
                    krw_cost = total_cost * exchange_rate
                    self.track_api_usage(
                        data["model"],
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        total_cost,
                        krw_cost,
                    )
                return summary
            else:
                logger.error(f"OpenAI API 오류: {response.status_code}")
                return f"요약 생성 중 오류 발생. 상태 코드: {response.status_code}"
        except Exception as e:
            logger.error(f"OpenAI API 요청 오류: {e}")
            return "요약 생성 중 오류 발생."

    def process_news_item(self, item: dict) -> dict:
        """
        뉴스 항목 처리: 기사 본문 추출 및 OpenAI 요약 추가
        """
        try:
            content, article_source = self.get_article_content(item["link"])
            if article_source and item["source"] == "Unknown":
                item["source"] = article_source
            item["ai_summary"] = self.summarize_with_openai(item["title"], content)
            return item
        except Exception as e:
            logger.error(f"뉴스 항목 처리 오류: {e}")
            return item

    def calculate_similarity_with_openai(self, news1: dict, news2: dict) -> float:
        """
        두 뉴스 제목의 의미적 유사도를 OpenAI API를 통해 정밀 비교
        """
        key = tuple(sorted([news1["title"], news2["title"]]))
        if key in self.similarity_cache:
            return self.similarity_cache[key]
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API 키가 설정되지 않았습니다.")
            return 0.0
        prompt = f"""
다음 두 뉴스 기사 제목의 의미적 유사도를 심층 분석해 주세요:

기사 1: {news1['title']}
기사 2: {news2['title']}

전처리된 제목:
기사 1: {self.preprocess_title(news1['title'])}
기사 2: {self.preprocess_title(news2['title'])}

단순히 동일한 단어가 얼마나 포함되어 있는지가 아니라, 두 제목이 같은 주제나 사건을 다루고 있는지의 의미론적 유사성에 중점을 두고 평가해 주세요.

다음 기준으로 유사도를 평가하세요:
1. 두 제목이 동일한 주요 사건/이슈를 다루고 있는가? (가장 중요)
2. 두 제목에서 언급된 주체(인물, 조직, 기관 등)가 동일한가?
3. 두 제목의 핵심 메시지나 관점이 유사한가?
4. 표현 방식만 다를 뿐 전달하는 정보가 본질적으로 동일한가?
5. 동일한 뉴스를 다른 관점에서 보도한 것인가?

0.0(완전히 다른 주제) ~ 1.0(거의 동일한 내용) 사이의 유사도 점수를 매겨주세요:
- 0.0~0.3: 완전히 다른 주제나 사건을 다룸
- 0.4~0.6: 관련된 주제지만 다른 측면이나 관점을 다룸
- 0.7~0.9: 같은 주제/사건을 다루며 유사한 정보 전달
- 0.9~1.0: 사실상 동일한 뉴스(다른 표현만 사용)

평가 후, 오직 0부터 1 사이의 소수점 둘째 자리 형식(예: 0.85)의 숫자만 응답해 주세요.
"""
        data = {
            "model": os.getenv("GPT_MODEL", "gpt-4o-mini"),
            "messages": [
                {
                    "role": "system",
                    "content": "당신은 뉴스 기사 유사성 분석 전문가입니다. 오직 숫자만 응답하세요.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 10,
        }
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json=data,
                timeout=20,
            )
            if response.status_code == 200:
                result = response.json()
                similarity_text = result["choices"][0]["message"]["content"].strip()
                match = re.search(r"(\d+\.\d{2}|\d+)", similarity_text)
                similarity = float(match.group(1)) if match else 0.0
                similarity = max(0.0, min(1.0, similarity))
                if "usage" in result:
                    prompt_tokens = result["usage"].get("prompt_tokens", 0)
                    completion_tokens = result["usage"].get("completion_tokens", 0)
                    total_tokens = result["usage"].get("total_tokens", 0)
                    model_prices = {
                        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
                        "gpt-4o": {"input": 5.0, "output": 15.0},
                        "gpt-4": {"input": 10.0, "output": 30.0},
                        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
                    }
                    price_info = model_prices.get(
                        data["model"], {"input": 0.15, "output": 0.60}
                    )
                    input_cost = (prompt_tokens / 1_000_000) * price_info["input"]
                    output_cost = (completion_tokens / 1_000_000) * price_info["output"]
                    total_cost = input_cost + output_cost
                    exchange_rate = float(os.getenv("USD_TO_KRW", 1350))
                    krw_cost = total_cost * exchange_rate
                    self.track_api_usage(
                        data["model"],
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        total_cost,
                        krw_cost,
                    )
                self.similarity_cache[key] = similarity
                # AI 유사도 결과를 로그에 출력
                logger.info(
                    f"비교: '{news1['title']}' vs '{news2['title']}' -> AI 유사도: {similarity:.2f}"
                )
                return similarity
            else:
                logger.error(f"유사도 계산 API 오류: {response.status_code}")
                return 0.0
        except Exception as e:
            logger.error(f"유사도 계산 중 오류 발생: {e}")
            return 0.0

    def filter_duplicate_news(
        self, news_items: list, similarity_threshold: float = 0.7
    ) -> list:
        """
        단순 Jaccard 유사도와 AI 정밀 비교를 결합해 중복 뉴스 항목을 제거하며,
        비교되는 기사 쌍마다 유사도를 로그에 출력합니다.
        """
        if not news_items or len(news_items) <= 1:
            return news_items

        unique_news = [news_items[0]]
        total_comparisons = len(news_items) - 1
        SIMPLE_SIM_HIGH = 0.6
        SIMPLE_SIM_LOW = 0.25

        with tqdm(
            total=total_comparisons,
            **get_tqdm_settings("중복 뉴스 필터링", COLORS["MAGENTA"], 3),
        ) as pbar:
            for i in range(1, len(news_items)):
                is_duplicate = False
                current_title = news_items[i]["title"]
                pbar.set_description(f"비교 중: {current_title[:30]}...")
                for unique_item in unique_news:
                    basic_sim = self.simple_jaccard_similarity(
                        current_title, unique_item["title"]
                    )
                    logger.info(
                        f"비교: '{current_title}' vs '{unique_item['title']}' -> 기본 유사도: {basic_sim:.2f}"
                    )
                    if basic_sim >= SIMPLE_SIM_HIGH:
                        is_duplicate = True
                        logger.info(
                            f"자동 중복 감지 (기본 유사도 {basic_sim:.2f}): {current_title}"
                        )
                        break
                    elif basic_sim <= SIMPLE_SIM_LOW:
                        continue
                    else:
                        ai_sim = self.calculate_similarity_with_openai(
                            news_items[i], unique_item
                        )
                        pbar.set_postfix_str(
                            f"자카드: {basic_sim:.2f} / AI: {ai_sim:.2f} / 기준: {similarity_threshold}"
                        )
                        if ai_sim >= similarity_threshold:
                            is_duplicate = True
                            logger.info(
                                f"AI 중복 감지: {current_title} (AI 유사도: {ai_sim:.2f})"
                            )
                            break
                if not is_duplicate:
                    unique_news.append(news_items[i])
                pbar.update(1)
        logger.info(f"중복 뉴스 제거: {len(news_items) - len(unique_news)}개 제거됨")
        return unique_news


# =============================================================================
# EmailService 클래스: Gmail API 및 SMTP+OAuth2 방식 이메일 발송
# =============================================================================
class EmailService:
    def __init__(self):
        self.sender_email: str = os.getenv("EMAIL_USERNAME")

    def get_gmail_service(self):
        SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
        creds = None
        token_json_path = os.getenv("EMAIL_OAUTH2_TOKEN", "token.json")

        if os.path.exists(token_json_path):
            try:
                logger.info("token.json에서 인증 정보 로드 중")
                with open(token_json_path, "r") as token_file:
                    token_data = json.load(token_file)
                    creds = Credentials(
                        token=token_data.get("token"),
                        refresh_token=token_data.get("refresh_token"),
                        token_uri=token_data.get(
                            "token_uri", "https://oauth2.googleapis.com/token"
                        ),
                        client_id=token_data.get("client_id"),
                        client_secret=token_data.get("client_secret"),
                        scopes=token_data.get("scopes", SCOPES),
                    )
            except Exception as e:
                logger.warning(f"token.json 로드 오류: {e}")
                creds = None

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("토큰 갱신 중...")
                creds.refresh(Request())
            else:
                logger.info("새 OAuth 인증 흐름 시작 중...")
                credentials_file = os.getenv(
                    "GMAIL_CREDENTIALS_FILE", "credentials.json"
                )
                if not os.path.exists(credentials_file):
                    logger.error(f"OAuth 인증 파일 없음: {credentials_file}")
                    return None
                try:
                    with open(credentials_file, "r") as f:
                        client_config = json.load(f)
                    if "installed" in client_config:
                        client_config["installed"]["redirect_uris"] = [
                            "urn:ietf:wg:oauth:2.0:oob"
                        ]
                    elif "web" in client_config:
                        client_config["web"]["redirect_uris"] = [
                            "urn:ietf:wg:oauth:2.0:oob"
                        ]
                    flow = InstalledAppFlow.from_client_config(
                        client_config, SCOPES, redirect_uri="urn:ietf:wg:oauth:2.0:oob"
                    )
                    auth_url, _ = flow.authorization_url(
                        access_type="offline", include_granted_scopes="true"
                    )
                    print("=" * 70)
                    print(
                        "Google 계정 인증이 필요합니다. 아래 URL을 브라우저에서 열고 인증 코드를 입력하세요:"
                    )
                    print("=" * 70)
                    print(auth_url)
                    print("=" * 70)
                    auth_code = input("인증 코드를 입력하세요: ").strip()
                    flow.fetch_token(code=auth_code)
                    creds = flow.credentials
                    print("인증 성공! 새 토큰이 저장되었습니다.")
                except Exception as e:
                    logger.error(f"OAuth 인증 오류: {e}")
                    return None
            try:
                logger.info("새 인증 정보를 token.json에 저장 중")
                token_data = {
                    "token": creds.token,
                    "refresh_token": creds.refresh_token,
                    "token_uri": creds.token_uri,
                    "client_id": creds.client_id,
                    "client_secret": creds.client_secret,
                    "scopes": list(creds.scopes),
                }
                with open(token_json_path, "w") as token_file:
                    json.dump(token_data, token_file)
            except Exception as e:
                logger.error(f"토큰 저장 오류: {e}")
        try:
            service = build("gmail", "v1", credentials=creds)
            return service
        except Exception as e:
            logger.error(f"Gmail 서비스 생성 오류: {e}")
            return None

    def send_email(
        self, recipient_name: str, recipient_email: str, subject: str, html_content: str
    ) -> bool:
        if not self.sender_email:
            logger.error("발신자 이메일이 설정되어 있지 않습니다.")
            return False
        try:
            service = self.get_gmail_service()
            if not service:
                logger.error("Gmail API 서비스 초기화 실패")
                return False
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.sender_email
            message["To"] = recipient_email
            message.attach(MIMEText(html_content, "html"))
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            sent_message = (
                service.users()
                .messages()
                .send(userId="me", body={"raw": raw_message})
                .execute()
            )
            logger.info(
                f"이메일 발송 성공: {recipient_email} (ID: {sent_message['id']})"
            )
            return True
        except Exception as e:
            logger.error(f"Gmail API 이메일 발송 오류: {e}")
            try:
                logger.warning("SMTP+OAuth2 방식으로 재시도...")
                oauth2_token = os.getenv("EMAIL_OAUTH2_TOKEN")
                if not oauth2_token:
                    logger.error("OAuth2 토큰이 설정되지 않았습니다.")
                    return False
                smtp_server = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
                smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "587"))
                import smtplib

                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()
                auth_string = (
                    f"user={self.sender_email}\1auth=Bearer {oauth2_token}\1\1"
                )
                server.docmd(
                    "AUTH", "XOAUTH2 " + base64.b64encode(auth_string.encode()).decode()
                )
                server.send_message(message)
                server.quit()
                logger.info(f"SMTP 방식 이메일 발송 성공: {recipient_email}")
                return True
            except Exception as smtp_error:
                logger.error(f"SMTP 이메일 발송 오류: {smtp_error}")
                return False


# =============================================================================
# SubscriberManager 클래스: 구독자 로드, 뉴스 수집 및 이메일 발송 담당
# =============================================================================
class SubscriberManager:
    def __init__(self, news_fetcher: NewsFetcher, email_service: EmailService):
        self.news_fetcher = news_fetcher
        self.email_service = email_service

    @staticmethod
    def load_subscribers(file_path: str = "subscribers.txt") -> list:
        subscribers = []
        if not os.path.exists(file_path):
            logger.error(f"구독자 파일이 존재하지 않습니다: {file_path}")
            return subscribers
        try:
            logger.info(f"구독자 파일 로드 중: {file_path}")
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                current_sub = {}
                for line in f:
                    line = line.strip()
                    if not line:
                        if current_sub.get("email") and current_sub.get("topics"):
                            subscribers.append(current_sub)
                        current_sub = {}
                        continue
                    if line.startswith("ID:"):
                        if current_sub.get("email") and current_sub.get("topics"):
                            subscribers.append(current_sub)
                        current_sub = {
                            "id": line[3:].strip(),
                            "name": "",
                            "email": "",
                            "topics": [],
                        }
                    elif line.startswith("이름:") and current_sub:
                        current_sub["name"] = line[3:].strip()
                    elif line.startswith("이메일:") and current_sub:
                        current_sub["email"] = line[4:].strip()
                    elif line.startswith("토픽:") and current_sub:
                        topics = [topic.strip() for topic in line[3:].split(",")]
                        current_sub["topics"] = topics
                if current_sub.get("email") and current_sub.get("topics"):
                    subscribers.append(current_sub)
            logger.info(f"총 {len(subscribers)}명의 구독자 로드 완료")
            return subscribers
        except Exception as e:
            logger.error(f"구독자 정보 로드 오류: {e}")
            return []

    @staticmethod
    def log_news_titles(news_items: list, prefix: str = "") -> None:
        if not news_items:
            logger.warning(f"{prefix} 뉴스 항목이 없습니다.")
            return
        logger.info(f"{prefix} 뉴스 제목 목록 (총 {len(news_items)}개):")
        for idx, item in enumerate(news_items, 1):
            topic = item.get("query", "기타")
            logger.info(f"{idx}. [{topic}] {item['title']}")

    def fetch_news_for_subscriber(self, subscriber: dict) -> list:
        """
        각 구독자의 관심 토픽에 대해 최대 30개의 최신 뉴스를 수집하고 처리
        중복 토픽은 한 번만 처리하여 이중 유사도 검색을 방지
        """
        collected_news = []
        topics = subscriber.get("topics", [])
        # 중복 토픽 제거
        unique_topics = list(dict.fromkeys(topics))
        name = subscriber.get("name", "구독자")

        # 토픽 중복이 있는 경우 로그에 기록
        if len(unique_topics) < len(topics):
            logger.info(
                f"{name}의 토픽 목록에서 {len(topics) - len(unique_topics)}개 중복 제거됨"
            )

        with tqdm(
            total=len(unique_topics),
            **get_tqdm_settings(f"{name}의 토픽 처리", COLORS["BLUE"], 1, True),
        ) as topic_bar:
            for topic in unique_topics:
                topic_bar.set_description(
                    f"{COLORS['CYAN']}토픽 '{topic}' 처리 중{COLORS['RESET']}"
                )
                logger.info(f"토픽 '{topic}' 뉴스 검색 중...")
                try:
                    news_items = self.news_fetcher.get_google_news(query=topic)
                    if news_items:
                        logger.info(f"토픽 '{topic}'에서 {len(news_items)}개 뉴스 발견")
                        unique_news = self.news_fetcher.filter_duplicate_news(
                            news_items, similarity_threshold=0.65
                        )
                        # 중복 제거 후 최대 30개 뉴스 중 처리
                        selected_news = unique_news
                        logger.info(
                            f"토픽 '{topic}'에서 {len(selected_news)}개 고유 뉴스 선정"
                        )
                        with ThreadPoolExecutor(max_workers=3) as executor:
                            futures = {
                                executor.submit(
                                    self.news_fetcher.process_news_item, item
                                ): item
                                for item in selected_news
                            }
                            with tqdm(
                                total=len(futures),
                                **get_tqdm_settings(
                                    f"'{topic}' 뉴스 처리", COLORS["GREEN"], 2
                                ),
                            ) as news_bar:
                                for future in as_completed(futures):
                                    try:
                                        processed_item = future.result()
                                        collected_news.append(processed_item)
                                        news_bar.update(1)
                                        news_bar.set_description(
                                            f"{COLORS['YELLOW']}처리: {processed_item['title'][:20]}...{COLORS['RESET']}"
                                        )
                                    except Exception as e:
                                        logger.error(f"뉴스 처리 중 오류: {e}")
                                        news_bar.update(1)
                    else:
                        logger.warning(f"토픽 '{topic}' 뉴스 없음")
                except Exception as e:
                    logger.error(f"토픽 '{topic}' 처리 중 오류: {e}")
                topic_bar.update(1)
        return collected_news

    def remove_duplicates_from_news(
        self, news_items: list, similarity_threshold: float = 0.65
    ) -> list:
        """
        수집된 뉴스에서 중복 항목을 제거하고, 각 토픽별로 최소 5개, 최대 7개의 뉴스가 확보되도록 보충
        그리고 각 비교 시 유사도 결과를 log에 출력함
        토픽 간 중복 검사를 효율적으로 처리하기 위해 글로벌 캐시 활용
        """
        if not news_items or len(news_items) <= 1:
            return news_items

        # 토픽별 그룹화
        topic_groups = {}
        for item in news_items:
            topic = item.get("query", "기타")
            topic_groups.setdefault(topic, []).append(item)

        # 모든 뉴스 아이템의 고유 ID 생성 (URL이나 제목 기반)
        news_id_map = {}
        for item in news_items:
            # 뉴스 아이템 식별자로 링크 사용
            news_id = item.get("link", item.get("title", ""))
            news_id_map[news_id] = item

        # 각 토픽 내에서 중복 제거 (캐시 사용으로 중복 검사 최적화)
        cleaned_by_topic = {}
        for topic, items in topic_groups.items():
            logger.info(f"토픽 '{topic}' 내부 중복 제거 시작 (총 {len(items)}개)")
            # 이미 처리된 아이템은 건너뛰기 위한 ID 세트
            processed_ids = set()
            unique_items = []

            for item in items:
                item_id = item.get("link", item.get("title", ""))
                if item_id in processed_ids:
                    continue

                unique_items.append(item)
                processed_ids.add(item_id)

            cleaned_items = self.news_fetcher.filter_duplicate_news(
                unique_items, similarity_threshold=0.7
            )
            cleaned_by_topic[topic] = cleaned_items
            logger.info(f"토픽 '{topic}': {len(cleaned_items)}개 남음")

        # 모든 토픽의 뉴스를 합치되, 이미 추가된 링크/제목은 제외
        all_news = []
        processed_ids = set()

        for items in cleaned_by_topic.values():
            for item in items:
                item_id = item.get("link", item.get("title", ""))
                if item_id not in processed_ids:
                    all_news.append(item)
                    processed_ids.add(item_id)

        # 최종 중복 제거에는 더 엄격한 기준 적용
        final_news = self.news_fetcher.filter_duplicate_news(
            all_news, similarity_threshold
        )

        # 각 토픽별로 최소 5개 뉴스 확보 (부족 시 후보 추가)
        for topic, items in cleaned_by_topic.items():
            count_in_final = sum(
                1 for item in final_news if item.get("query", "기타") == topic
            )
            if count_in_final < 5:
                needed = 5 - count_in_final
                logger.info(f"토픽 '{topic}'에 추가 {needed}개 뉴스 필요")
                # 아직 추가되지 않은 후보만 추가
                candidates = []
                for item in items:
                    if item not in final_news:
                        item_id = item.get("link", item.get("title", ""))
                        if not any(
                            i.get("link", "") == item_id
                            or i.get("title", "") == item_id
                            for i in final_news
                        ):
                            candidates.append(item)
                final_news.extend(candidates[:needed])

        # 최종적으로 각 토픽은 최대 7개로 제한
        topic_final = {}
        for item in final_news:
            topic = item.get("query", "기타")
            topic_final.setdefault(topic, []).append(item)
        final_news_clipped = []
        for topic, items in topic_final.items():
            if len(items) > 7:
                final_news_clipped.extend(items[:7])
            else:
                final_news_clipped.extend(items)
        logger.info(f"최종 뉴스 항목: {len(final_news_clipped)}개")
        return final_news_clipped

    def process_subscriber(self, subscriber: dict, collected_news: list) -> dict:
        name = subscriber.get("name", "구독자")
        email_addr = subscriber.get("email", "")
        topics = subscriber.get("topics", [])
        if not topics:
            logger.warning(f"{name}의 관심 토픽이 없습니다.")
            return {
                "name": name,
                "email": email_addr,
                "success": False,
                "reason": "관심 토픽 없음",
            }
        if not collected_news:
            logger.warning(f"{name}의 뉴스 수집 결과가 없습니다.")
            return {
                "name": name,
                "email": email_addr,
                "success": False,
                "reason": "뉴스 없음",
            }

        logger.info(f"{name} 이메일 콘텐츠 준비 (수집 뉴스: {len(collected_news)}개)")
        deduped_news = self.remove_duplicates_from_news(
            collected_news, similarity_threshold=0.65
        )
        self.log_news_titles(deduped_news, prefix=f"{name} 최종 뉴스 목록")

        # 그룹화하여 토픽별 뉴스 리스트 구성
        # 중복 토픽들을 모두 고려하여 각 토픽별로 적절한 뉴스를 배치
        topic_news = {}
        for item in deduped_news:
            item_topic = item.get("query", "기타")
            # 구독자의 모든 토픽에 대해 체크
            for topic in topics:
                # 뉴스 아이템의 쿼리와 토픽이 일치하면 해당 토픽에 추가
                if topic == item_topic:
                    topic_news.setdefault(topic, []).append(item)

        if not topic_news:
            logger.warning(f"{name}의 관심 토픽과 일치하는 뉴스가 없습니다.")
            return {
                "name": name,
                "email": email_addr,
                "success": False,
                "reason": "관심 뉴스 없음",
            }

        # 각 토픽별로 출력되는 뉴스는 최대 7개만 사용 (최소 5개는 remove_duplicates에서 보충됨)
        for topic in topic_news:
            if len(topic_news[topic]) > 7:
                topic_news[topic] = topic_news[topic][:7]

        current_date = datetime.now().strftime("%Y년 %m월 %d일")
        html_content = self.build_email_html(name, current_date, topic_news)
        subject = f"{name}님을 위한 관심 토픽 뉴스 요약 ({current_date})"
        success = self.email_service.send_email(name, email_addr, subject, html_content)
        return {
            "name": name,
            "email": email_addr,
            "success": success,
            "news_count": len(collected_news),
        }

    @staticmethod
    def build_email_html(name: str, current_date: str, topic_news: dict) -> str:
        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{name}님을 위한 맞춤 뉴스 요약</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'Noto Sans KR', sans-serif; color: #1f2937; line-height: 1.5; background-color: #f8fafc; }}
  </style>
</head>
<body>
  <div style="background: linear-gradient(to right, #2563eb, #3b82f6); padding: 24px; text-align: center; color: white;">
    <h1 style="font-size: 24px; font-weight: 700;">{name}님을 위한 맞춤 뉴스</h1>
    <p style="font-size: 16px;">{current_date} 발행</p>
  </div>
  <div style="max-width: 700px; margin: 0 auto; padding: 24px; background: white; border-radius: 8px;">
    <div style="margin-bottom: 24px; padding: 16px; background: #f0f9ff; border-left: 4px solid #3b82f6;">
      <p style="color: #1e40af; font-size: 16px;">안녕하세요, <strong>{name}</strong>님!</p>
      <p style="color: #1e40af; font-size: 14px; margin-top: 8px;">최근 관심 토픽 뉴스를 정리했습니다.</p>
    </div>
"""
        count = 0
        for topic, news_list in topic_news.items():
            count += 1
            background = "#f8fafc" if count % 2 == 0 else "#ffffff"
            html += f"""
    <div style="margin-bottom: 32px; padding-bottom: 24px; border-bottom: 1px solid #e5e7eb;">
      <div style="display: flex; align-items: center; margin-bottom: 16px;">
        <div style="background: #3b82f6; border-radius: 9999px; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; margin-right: 8px;">
          <span style="color: white; font-weight: bold;">{count}</span>
        </div>
        <h2 style="color: #1e40af; font-size: 20px; font-weight: 600;">{topic}</h2>
      </div>
"""
            if not news_list:
                html += f"""
      <div style="margin-bottom: 24px; padding: 16px; background: {background}; border: 1px solid #e5e7eb; border-radius: 8px; text-align: center;">
        <p style="color: #6b7280; font-size: 14px;">관련 뉴스 기사가 없습니다.</p>
      </div>
"""
            else:
                for item in news_list:
                    summary = item.get(
                        "ai_summary", item.get("summary", "요약 정보가 없습니다.")
                    )
                    summary = summary if len(summary) <= 300 else summary[:300]
                    html += f"""
      <div style="margin-bottom: 24px; padding: 16px; background: {background}; border: 1px solid #e5e7eb; border-radius: 8px;">
        <h3 style="font-size: 18px; font-weight: 600; color: #1e3a8a; margin-bottom: 8px;">
          <a href="{item['link']}" target="_blank" style="text-decoration: none; color: inherit;">{item['title']}</a>
        </h3>
        <div style="font-size: 13px; color: #6b7280; margin-bottom: 12px;">
          <span>{item['source']}</span> | <span>{item['published']}</span>
        </div>
        <div style="font-size: 14px; line-height: 1.6; color: #4b5563; margin-bottom: 16px;">{summary}</div>
      </div>
"""
            html += "    </div>\n"
        html += f"""
    <div style="text-align: center; font-size: 12px; color: #6b7280; margin-top: 32px; border-top: 1px solid #e5e7eb; padding-top: 16px;">
      <p>이 이메일은 자동으로 발송되었습니다.</p>
      <p>© {datetime.now().year} 뉴스레터 서비스 | {current_date}</p>
    </div>
  </div>
  <div style="background: #1e40af; padding: 16px; text-align: center; color: white; font-size: 12px; margin-top: 24px;">
    <p>구독 관련 문의는 관리자에게 연락하세요.</p>
    <p>이 메일은 발신 전용입니다.</p>
  </div>
</body>
</html>
"""
        return html


# =============================================================================
# 메인 실행 함수
# =============================================================================
def main() -> None:
    try:
        logger.info("뉴스 크롤링 및 이메일 발송 시작")
        news_fetcher = NewsFetcher()
        email_service = EmailService()
        subscriber_manager = SubscriberManager(news_fetcher, email_service)
        subscribers = subscriber_manager.load_subscribers()
        if not subscribers:
            logger.error("구독자가 없습니다. subscribers.txt 파일을 확인하세요.")
            return

        # 모든 구독자의 고유 토픽 목록 확인
        all_topics = set()
        for subscriber in subscribers:
            topics = subscriber.get("topics", [])
            all_topics.update(topics)
        logger.info(
            f"전체 {len(subscribers)}명 구독자, 고유 토픽 {len(all_topics)}개 발견"
        )

        # 토픽별 뉴스 캐시 구성 (한 토픽은 한 번만 처리)
        topic_news_cache = {}

        with tqdm(
            total=len(subscribers),
            **get_tqdm_settings("전체 구독자 처리", COLORS["CYAN"], 0, True),
        ) as main_bar:
            for subscriber in subscribers:
                name = subscriber.get("name", "Unknown")
                email_addr = subscriber.get("email", "")
                topics = subscriber.get("topics", [])
                main_bar.set_description(
                    f"{COLORS['BLUE']}구독자 '{name}' 처리 중{COLORS['RESET']}"
                )
                if not topics or not email_addr:
                    logger.warning(
                        f"{name}의 관심 토픽 또는 이메일 주소가 없습니다. 건너뜁니다."
                    )
                    main_bar.update(1)
                    continue

                logger.info(f"{name} 뉴스 수집 시작")
                # 이미 캐시된 토픽 뉴스가 있으면 재사용
                cached_subscriber_news = []
                topics_to_fetch = []

                for topic in topics:
                    if topic in topic_news_cache:
                        logger.info(
                            f"캐시에서 토픽 '{topic}'의 뉴스 {len(topic_news_cache[topic])}개 로드"
                        )
                        cached_subscriber_news.extend(topic_news_cache[topic])
                    else:
                        topics_to_fetch.append(topic)

                # 캐시에 없는 토픽만 새로 처리
                if topics_to_fetch:
                    # 임시로 캐시에 없는 토픽만 담긴 구독자 객체 생성
                    temp_subscriber = {
                        "name": name,
                        "email": email_addr,
                        "topics": topics_to_fetch,
                    }
                    new_news = subscriber_manager.fetch_news_for_subscriber(
                        temp_subscriber
                    )

                    # 처리된 새 뉴스를 토픽별로 캐시에 저장
                    for item in new_news:
                        topic = item.get("query", "기타")
                        if topic in topics_to_fetch:
                            topic_news_cache.setdefault(topic, []).append(item)

                    cached_subscriber_news.extend(new_news)

                # 전체 구독자 뉴스 (캐시 + 새로 가져온 뉴스)
                subscriber_news = cached_subscriber_news

                if subscriber_news:
                    logger.info(f"{name}의 뉴스 수집 완료: {len(subscriber_news)}개")
                    subscriber_manager.log_news_titles(
                        subscriber_news, prefix=f"{name} 초기 수집"
                    )
                    result = subscriber_manager.process_subscriber(
                        subscriber, subscriber_news
                    )
                    if result.get("success"):
                        logger.info(f"{name}에게 이메일 발송 성공")
                        main_bar.set_postfix_str(
                            f"뉴스: {len(subscriber_news)} / 발송: 성공"
                        )
                    else:
                        logger.error(
                            f"{name} 이메일 발송 실패: {result.get('reason', '알 수 없음')}"
                        )
                        main_bar.set_postfix_str(
                            f"뉴스: {len(subscriber_news)} / 발송: 실패"
                        )
                else:
                    logger.warning(f"{name}의 관심 토픽 뉴스가 없습니다.")
                    main_bar.set_postfix_str("뉴스: 0")
                time.sleep(1)
                main_bar.update(1)

        # 토큰 사용량 및 API 통계 로그 출력
        stats = news_fetcher.api_usage_stats

        # 로그 파일에 통계 기록
        logger.info("=" * 50)
        logger.info("API 사용량 통계")
        logger.info("=" * 50)
        logger.info(f"총 API 호출 횟수: {stats['api_calls']}회")
        logger.info(f"총 사용 토큰: {stats['total_tokens']:,}개")
        logger.info(f"  - 프롬프트 토큰: {stats['prompt_tokens']:,}개")
        logger.info(f"  - 응답 토큰: {stats['completion_tokens']:,}개")
        logger.info(
            f"총 비용: ${stats['total_cost_usd']:.4f} (약 ₩{stats['total_cost_krw']:.0f})"
        )

        # 모델별 사용량 출력
        logger.info("-" * 50)
        logger.info("모델별 사용량:")
        for model, model_stats in stats["models"].items():
            logger.info(f"  ▶ {model}:")
            logger.info(f"    - API 호출: {model_stats['api_calls']}회")
            logger.info(f"    - 총 토큰: {model_stats['total_tokens']:,}개")
            logger.info(
                f"    - 비용: ${model_stats['total_cost_usd']:.4f} (약 ₩{model_stats['total_cost_krw']:.0f})"
            )

        logger.info("=" * 50)
        logger.info("모든 구독자 처리 완료")

        # 터미널에도 통계 출력
        print("\n")
        print(f"{COLORS['CYAN']}{'=' * 60}{COLORS['RESET']}")
        print(f"{COLORS['CYAN']}💰 API 사용량 통계 요약{COLORS['RESET']}")
        print(f"{COLORS['CYAN']}{'=' * 60}{COLORS['RESET']}")
        print(
            f"{COLORS['WHITE']}📊 총 API 호출: {COLORS['YELLOW']}{stats['api_calls']:,}회{COLORS['RESET']}"
        )
        print(
            f"{COLORS['WHITE']}🔤 총 사용 토큰: {COLORS['YELLOW']}{stats['total_tokens']:,}개{COLORS['RESET']}"
        )
        print(
            f"{COLORS['WHITE']}  - 입력 토큰: {COLORS['GREEN']}{stats['prompt_tokens']:,}개{COLORS['RESET']}"
        )
        print(
            f"{COLORS['WHITE']}  - 출력 토큰: {COLORS['GREEN']}{stats['completion_tokens']:,}개{COLORS['RESET']}"
        )
        print(
            f"{COLORS['WHITE']}💵 총 비용: {COLORS['MAGENTA']}${stats['total_cost_usd']:.4f} (약 ₩{stats['total_cost_krw']:.0f}){COLORS['RESET']}"
        )

        print(f"{COLORS['CYAN']}{'-' * 60}{COLORS['RESET']}")
        print(f"{COLORS['CYAN']}📋 모델별 사용량:{COLORS['RESET']}")

        for model, model_stats in stats["models"].items():
            print(f"{COLORS['BLUE']}  ▶ {model}:{COLORS['RESET']}")
            print(
                f"{COLORS['WHITE']}    - API 호출: {COLORS['YELLOW']}{model_stats['api_calls']}회{COLORS['RESET']}"
            )
            print(
                f"{COLORS['WHITE']}    - 총 토큰: {COLORS['YELLOW']}{model_stats['total_tokens']:,}개{COLORS['RESET']}"
            )
            print(
                f"{COLORS['WHITE']}    - 비용: {COLORS['MAGENTA']}${model_stats['total_cost_usd']:.4f} (약 ₩{model_stats['total_cost_krw']:.0f}){COLORS['RESET']}"
            )

        print(f"{COLORS['CYAN']}{'=' * 60}{COLORS['RESET']}")
        print(f"\n{COLORS['GREEN']}✅ 모든 처리가 완료되었습니다!{COLORS['RESET']}")

    except Exception as e:
        logger.error(f"메인 실행 중 오류 발생: {e}")
    finally:
        colorama.deinit()


if __name__ == "__main__":
    main()
