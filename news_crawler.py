#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import smtplib
import logging
import pathlib
import time
import base64
import pickle
import requests
import xml.etree.ElementTree as ET
import re
import urllib.parse
import email.utils
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import concurrent.futures
from tqdm import tqdm
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# 환경변수 로드
load_dotenv()

# --- 로깅 설정 ---
pathlib.Path("logs").mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/news_crawler_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_file, mode="a", encoding="utf-8", errors="replace"),
    ],
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("news_crawler")


def log_to_file(message, level="info"):
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    else:
        logger.debug(message)


# --- 전역 API 사용량 통계 ---
api_usage_stats = {
    "total_tokens": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_cost_usd": 0.0,
    "total_cost_krw": 0.0,
    "api_calls": 0,
    "models": {},
}


def track_api_usage(
    model, prompt_tokens, completion_tokens, total_tokens, usd_cost, krw_cost
):
    global api_usage_stats
    api_usage_stats["total_tokens"] += total_tokens
    api_usage_stats["prompt_tokens"] += prompt_tokens
    api_usage_stats["completion_tokens"] += completion_tokens
    api_usage_stats["total_cost_usd"] += usd_cost
    api_usage_stats["total_cost_krw"] += krw_cost
    api_usage_stats["api_calls"] += 1

    if model not in api_usage_stats["models"]:
        api_usage_stats["models"][model] = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost_usd": 0.0,
            "total_cost_krw": 0.0,
            "api_calls": 0,
        }
    api_usage_stats["models"][model]["total_tokens"] += total_tokens
    api_usage_stats["models"][model]["prompt_tokens"] += prompt_tokens
    api_usage_stats["models"][model]["completion_tokens"] += completion_tokens
    api_usage_stats["models"][model]["total_cost_usd"] += usd_cost
    api_usage_stats["models"][model]["total_cost_krw"] += krw_cost
    api_usage_stats["models"][model]["api_calls"] += 1


# --- 뉴스 크롤러 기능 ---


def get_google_news(query="", country="kr", language="ko"):
    """
    최근 2일 이내의 Google News RSS 피드 뉴스를 가져옵니다.
    """
    base_url = "https://news.google.com/rss"
    if query:
        encoded_query = urllib.parse.quote(query)
        url = f"{base_url}/search?q={encoded_query}"
    else:
        url = base_url
    url += f"&hl={language}-{country}&gl={country}&ceid={country}:{language}"

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            log_to_file(f"Error: HTTP 상태 코드 {response.status_code}", "error")
            return []
        root = ET.fromstring(response.content)
        news_items = []
        two_days_ago = datetime.now(timezone.utc) - timedelta(days=2)
        for item in root.findall(".//item"):
            title_elem = item.find("title")
            link_elem = item.find("link")
            pub_date_elem = item.find("pubDate")
            description_elem = item.find("description")
            formatted_date = "No date info"
            is_recent = True
            parsed_date = None
            if pub_date_elem is not None and pub_date_elem.text:
                try:
                    parsed_date = email.utils.parsedate_to_datetime(pub_date_elem.text)
                    formatted_date = parsed_date.strftime("%Y-%m-%d %H:%M")
                    is_recent = parsed_date >= two_days_ago
                except Exception:
                    log_to_file("날짜 파싱 오류", "warning")
                    formatted_date = pub_date_elem.text
            if not is_recent:
                continue
            source = None
            if title_elem is not None and title_elem.text:
                title_source_match = re.search(
                    r"^(.*?)\s*-\s*([^-]+)$", title_elem.text
                )
                if title_source_match:
                    source = title_source_match.group(2).strip()
            if (
                source is None
                and description_elem is not None
                and description_elem.text
            ):
                source_match1 = re.search(
                    r'<font size="-1">([^<]+)</font>', description_elem.text
                )
                if source_match1:
                    source = source_match1.group(1).strip()
                else:
                    source_match2 = re.search(r"<b>([^<]+)</b>", description_elem.text)
                    if source_match2:
                        source = source_match2.group(1).strip()
            if source is None:
                source_elem = item.find("source")
                if source_elem is not None and source_elem.text:
                    source = source_elem.text.strip()
            news_item = {
                "title": title_elem.text if title_elem is not None else "No Title",
                "link": link_elem.text if link_elem is not None else "#",
                "published": formatted_date,
                "published_raw": (
                    pub_date_elem.text if pub_date_elem is not None else None
                ),
                "published_datetime": parsed_date,
                "source": source if source else "Unknown",
                "query": query,
            }
            if description_elem is not None:
                summary = re.sub(r"<[^>]+>", "", description_elem.text)
                news_item["summary"] = summary.strip()
            news_items.append(news_item)
        return news_items
    except Exception as e:
        log_to_file(f"뉴스 가져오기 오류: {e}", "error")
        return []


def get_article_content(url):
    """
    뉴스 기사 URL에서 본문 내용을 추출합니다.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            log_to_file(f"Error: HTTP 상태 코드 {response.status_code}", "error")
            return "기사 내용을 가져올 수 없습니다.", None
        soup = BeautifulSoup(response.content, "html.parser")
        source = None
        meta_site_name = soup.find("meta", property="og:site_name")
        if meta_site_name:
            source = meta_site_name.get("content")
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
            main_content_tags = soup.find_all(
                ["p", "div"], class_=re.compile(r"article|content|body|text|main", re.I)
            )
            content = " ".join(
                tag.get_text(strip=True)
                for tag in main_content_tags
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
            return content[:2000] + "..." if len(content) > 2000 else content, source
        return "기사 내용을 추출할 수 없습니다.", source
    except Exception as e:
        log_to_file(f"기사 내용 가져오기 오류: {e}", "error")
        return "기사 내용을 가져오는 중 오류가 발생했습니다.", None


def summarize_with_openai(title, text):
    """
    OpenAI API를 이용해 기사 내용을 요약합니다.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log_to_file("OpenAI API 키가 설정되지 않았습니다.", "error")
        return "요약을 생성할 수 없습니다. API 키를 확인하세요."
    try:
        system_prompt = """당신은 최고의 뉴스 에디터로, 뉴스 기사를 명확하고 풍부하게 요약하는 전문가입니다.

뉴스 기사의 핵심 내용을 파악하여 통합된 하나의 요약문을 작성해주세요. 이 요약문은 다음 내용을 포함해야 합니다:

- 기사의 핵심 주제와 가장 중요한 정보
- 주요 사실, 관련 데이터, 중요한 인용구
- 해당 뉴스의 맥락적 중요성과 잠재적 영향

요약 작성 시 다음 원칙을 따르세요:
- 정확성: 원문의 핵심 정보와 맥락을 정확하게 전달
- 간결성: 불필요한 정보 제외, 핵심만 포함
- 객관성: 개인적 의견이나 편향 없이 중립적 서술
- 가독성: 명확하고 이해하기 쉬운 언어 사용
- 완전성: 주요 질문(누가, 무엇을, 언제, 어디서, 왜, 어떻게)에 대한 답변 포함

문단 구분은 자연스럽게 하되, 번호나 글머리 기호로 구분하지 말고 하나의 흐름있는 요약문으로 작성하세요. 전체 요약 길이는 500자 내외로 작성해주세요."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        model_name = os.getenv("GPT_MODEL", "gpt-4o-mini")
        data = {
            "model": model_name,
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
        response = requests.post(url, headers=headers, json=data, timeout=30)
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
                default_price = {"input": 0.15, "output": 0.60}
                price_info = model_prices.get(model_name, default_price)
                input_cost = (prompt_tokens / 1000000) * price_info["input"]
                output_cost = (completion_tokens / 1000000) * price_info["output"]
                total_cost = input_cost + output_cost
                exchange_rate = float(os.getenv("USD_TO_KRW", 1350))
                krw_cost = total_cost * exchange_rate
                track_api_usage(
                    model_name,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    total_cost,
                    krw_cost,
                )
            return summary
        else:
            log_to_file(f"OpenAI API 오류: {response.status_code}", "error")
            error_detail = response.json() if response.content else "상세 정보 없음"
            log_to_file(f"오류 상세: {error_detail}", "error")
            return (
                f"요약 생성 중 오류가 발생했습니다. 상태 코드: {response.status_code}"
            )
    except Exception as e:
        log_to_file(f"OpenAI API 요청 오류: {e}", "error")
        return "요약 생성 중 오류가 발생했습니다."


def process_news_item(item):
    """
    기사 URL에서 본문을 가져오고 요약한 후 뉴스 항목에 추가합니다.
    """
    try:
        article_content, article_source = get_article_content(item["link"])
        if article_source and item["source"] == "Unknown":
            item["source"] = article_source
        ai_summary = summarize_with_openai(item["title"], article_content)
        item["ai_summary"] = ai_summary
        return item
    except Exception as e:
        log_to_file(f"뉴스 항목 처리 오류: {e}", "error")
        return item


def load_subscribers(file_path="subscribers.txt"):
    """
    구독자 정보를 텍스트 파일에서 로드합니다.
    """
    subscribers = []
    if not os.path.exists(file_path):
        log_to_file(f"구독자 파일이 존재하지 않습니다: {file_path}", "error")
        return subscribers
    try:
        log_to_file(f"구독자 파일 로드 중: {file_path}", "info")
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            current_subscriber = None
            for line in f:
                line = line.strip()
                if not line:
                    if (
                        current_subscriber
                        and current_subscriber.get("email")
                        and current_subscriber.get("topics")
                    ):
                        subscribers.append(current_subscriber)
                        current_subscriber = None
                    continue
                if line.startswith("ID:"):
                    if (
                        current_subscriber
                        and current_subscriber.get("email")
                        and current_subscriber.get("topics")
                    ):
                        subscribers.append(current_subscriber)
                    current_subscriber = {
                        "id": line[3:].strip(),
                        "name": "",
                        "email": "",
                        "topics": [],
                    }
                elif line.startswith("이름:") and current_subscriber:
                    current_subscriber["name"] = line[3:].strip()
                elif line.startswith("이메일:") and current_subscriber:
                    current_subscriber["email"] = line[4:].strip()
                elif line.startswith("토픽:") and current_subscriber:
                    topics = [topic.strip() for topic in line[3:].split(",")]
                    current_subscriber["topics"] = topics
            if (
                current_subscriber
                and current_subscriber.get("email")
                and current_subscriber.get("topics")
            ):
                subscribers.append(current_subscriber)
        for sub in subscribers:
            log_to_file(
                f"구독자 로드됨: {sub['name']} ({sub['email']}), 토픽: {sub['topics']}",
                "info",
            )
        log_to_file(f"총 {len(subscribers)}명의 구독자 로드 완료", "info")
        return subscribers
    except Exception as e:
        log_to_file(f"구독자 정보 로드 오류: {e}", "error")
        return []


def fetch_subscriber_news(subscriber):
    """
    구독자의 관심 토픽별로 뉴스 항목을 수집 및 처리합니다.
    중복 코드를 제거하기 위해 별도의 함수로 분리하였습니다.
    """
    collected_news_items = []
    topics = subscriber.get("topics", [])
    for topic in tqdm(
        topics, desc=f"{subscriber.get('name', '구독자')}'s topics", ncols=80
    ):
        log_to_file(f"토픽 '{topic}'에 대한 뉴스 검색 중...", "info")
        try:
            news_items = get_google_news(query=topic)
            if news_items:
                log_to_file(
                    f"토픽 '{topic}'에서 {len(news_items)}개의 뉴스 항목 발견", "info"
                )
                selected_news = news_items[:3]
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = {
                        executor.submit(process_news_item, item): item
                        for item in selected_news
                    }
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            processed_item = future.result()
                            collected_news_items.append(processed_item)
                        except Exception as e:
                            log_to_file(f"뉴스 항목 처리 중 오류 발생: {e}", "error")
            else:
                log_to_file(f"토픽 '{topic}'에 대한 뉴스를 찾지 못했습니다.", "warning")
        except Exception as e:
            log_to_file(f"토픽 '{topic}' 처리 중 오류 발생: {e}", "error")
    return collected_news_items


def process_subscriber(subscriber, collected_news_items):
    """
    수집된 뉴스 항목을 구독자의 관심 토픽별로 분류하여 이메일 HTML 콘텐츠를 생성한 후,
    이메일 발송을 수행합니다.
    """
    try:
        name = subscriber["name"]
        email = subscriber["email"]
        topics = subscriber["topics"]

        if not topics:
            log_to_file(f"구독자 {name}의 관심 토픽이 없습니다.", "warning")
            return {
                "name": name,
                "email": email,
                "success": False,
                "reason": "관심 토픽이 없습니다.",
            }
        if not collected_news_items:
            log_to_file(f"구독자 {name}의 관심 토픽에 대한 뉴스가 없습니다.", "warning")
            return {
                "name": name,
                "email": email,
                "success": False,
                "reason": "관심 토픽 뉴스 없음.",
            }

        log_to_file(
            f"구독자 {name}의 이메일 콘텐츠 준비 중. (총 뉴스 항목: {len(collected_news_items)})",
            "info",
        )
        topic_news = {}
        for item in collected_news_items:
            topic = item.get("query", "기타")
            if topic in topics:
                topic_news.setdefault(topic, []).append(item)
                log_to_file(
                    f"[{topic}] {item['title']} (출처: {item['source']}, 게시일: {item['published']})",
                    "info",
                )
        if not topic_news:
            log_to_file(
                f"구독자 {name}의 관심 토픽과 일치하는 뉴스가 없습니다.", "warning"
            )
            return {
                "name": name,
                "email": email,
                "success": False,
                "reason": "관심 토픽 뉴스 없음.",
            }

        current_date = datetime.now().strftime("%Y년 %m월 %d일")
        html_content = f"""<!DOCTYPE html>
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
        topic_count = 0
        for topic, news_list in topic_news.items():
            topic_count += 1
            background_color = "#f8fafc" if topic_count % 2 == 0 else "#ffffff"
            html_content += f"""
        <div style="margin-bottom: 32px; padding-bottom: 24px; border-bottom: 1px solid #e5e7eb;">
            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                <div style="background: #3b82f6; border-radius: 9999px; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; margin-right: 8px;">
                    <span style="color: white; font-weight: bold;">{topic_count}</span>
                </div>
                <h2 style="color: #1e40af; font-size: 20px; font-weight: 600;">{topic}</h2>
            </div>
"""
            for item in news_list:
                summary = item.get(
                    "ai_summary", item.get("summary", "요약 정보가 없습니다.")
                )
                if len(summary) > 300:
                    summary = summary[:297] + "..."
                html_content += f"""
            <div style="margin-bottom: 24px; padding: 16px; background: {background_color}; border: 1px solid #e5e7eb; border-radius: 8px;">
                <h3 style="font-size: 18px; font-weight: 600; color: #1e3a8a; margin-bottom: 8px;">{item['title']}</h3>
                <div style="font-size: 13px; color: #6b7280; margin-bottom: 12px;">
                    <span>{item['source']}</span> | <span>{item['published']}</span>
                </div>
                <div style="font-size: 14px; line-height: 1.6; color: #4b5563; margin-bottom: 16px;">{summary}</div>
                <a href="{item['link']}" style="padding: 8px 16px; background: #3b82f6; color: white; text-decoration: none; border-radius: 4px;">기사 원문 보기</a>
            </div>
"""
            html_content += """
        </div>
"""
        html_content += f"""
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
        subject = f"{name}님을 위한 관심 토픽 뉴스 요약 ({current_date})"
        success = send_email(name, email, subject, html_content)
        if success:
            log_to_file(
                f"{name}님에게 {len(collected_news_items)}개의 뉴스가 포함된 이메일 발송 성공",
                "info",
            )
        else:
            log_to_file(f"{name}님에게 이메일 발송 실패", "error")
        return {
            "name": name,
            "email": email,
            "success": success,
            "news_count": len(collected_news_items),
        }
    except Exception as e:
        log_to_file(
            f"구독자 처리 오류 ({subscriber.get('name', 'Unknown')}): {e}", "error"
        )
        return {
            "name": subscriber.get("name", "Unknown"),
            "email": subscriber.get("email", "Unknown"),
            "success": False,
            "reason": str(e),
        }


# --- Gmail 및 이메일 발송 기능 ---


def get_gmail_service():
    SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
    creds = None

    # 토큰 파일 경로
    token_json_path = os.getenv("EMAIL_OAUTH2_TOKEN", "token.json")
    pickle_token_path = "token.pickle"  # 기존 pickle 파일 경로 (하위 호환성 유지)

    # 기존 token.json 파일이 있으면 먼저 시도
    if os.path.exists(token_json_path):
        try:
            log_to_file(f"token.json 파일에서 인증 정보 로드 중", "info")
            with open(token_json_path, "r") as token_file:
                import json

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
            log_to_file(f"token.json 파일 로드 중 오류 발생: {e}", "warning")
            creds = None

    # 기존 pickle 파일 시도 (하위 호환성)
    if creds is None and os.path.exists(pickle_token_path):
        try:
            log_to_file("token.pickle 파일에서 인증 정보 로드 중 (하위 호환성)", "info")
            with open(pickle_token_path, "rb") as token:
                creds = pickle.load(token)
        except Exception as e:
            log_to_file(f"token.pickle 파일 로드 중 오류 발생: {e}", "warning")
            creds = None

    # 인증 정보가 없거나 유효하지 않은 경우 새로 인증
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            log_to_file("토큰 갱신 중...", "info")
            creds.refresh(Request())
        else:
            log_to_file("새 OAuth 인증 흐름 시작 중...", "info")
            credentials_file = os.getenv("GMAIL_CREDENTIALS_FILE", "credentials.json")
            if not os.path.exists(credentials_file):
                log_to_file(
                    f"OAuth 인증 정보 파일이 없습니다: {credentials_file}", "error"
                )
                return None

            try:
                # InstalledAppFlow 설정 (콘솔 인증 방식)
                # redirect_uri를 명시적으로 'urn:ietf:wg:oauth:2.0:oob'로 설정하는 커스텀 JSON 로드
                with open(credentials_file, "r") as f:
                    import json

                    client_config = json.load(f)

                # 리디렉션 URI 설정 추가
                if "installed" in client_config:
                    client_config["installed"]["redirect_uris"] = [
                        "urn:ietf:wg:oauth:2.0:oob"
                    ]
                elif "web" in client_config:
                    client_config["web"]["redirect_uris"] = [
                        "urn:ietf:wg:oauth:2.0:oob"
                    ]

                # 수정된 설정으로 InstalledAppFlow 생성
                flow = InstalledAppFlow.from_client_config(
                    client_config, SCOPES, redirect_uri="urn:ietf:wg:oauth:2.0:oob"
                )

                # 인증 URL 생성 및 사용자 안내
                auth_url, _ = flow.authorization_url(
                    access_type="offline",
                    include_granted_scopes="true",
                )

                print("\n" + "=" * 70)
                print("Google 계정 인증이 필요합니다.")
                print(
                    "아래 URL을 복사하여 웹 브라우저에서 열고 Google 계정으로 로그인하세요:"
                )
                print("=" * 70)
                print(f"\n{auth_url}\n")
                print("=" * 70)

                # 사용자로부터 인증 코드 입력 받기
                auth_code = input(
                    "\n브라우저에서 인증 후 받은 코드를 붙여넣고 Enter 키를 누르세요: "
                ).strip()

                # 입력받은 코드로 토큰 발급
                flow.fetch_token(code=auth_code)
                creds = flow.credentials
                print("\n인증 성공! 토큰이 저장되었습니다.\n")

            except Exception as e:
                log_to_file(f"인증 코드 처리 중 오류 발생: {e}", "error")
                print(f"\n인증 실패: {e}\n")
                print("자세한 오류 정보:")
                import traceback

                traceback.print_exc()
                return None

        # token.json에 저장
        try:
            log_to_file("새 인증 정보를 token.json에 저장 중", "info")
            token_data = {
                "token": creds.token,
                "refresh_token": creds.refresh_token,
                "token_uri": creds.token_uri,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "scopes": creds.scopes,
            }
            with open(token_json_path, "w") as token_file:
                import json

                json.dump(token_data, token_file)

            # 하위 호환성을 위해 pickle 파일도 유지 (선택적)
            with open(pickle_token_path, "wb") as token:
                pickle.dump(creds, token)
        except Exception as e:
            log_to_file(f"인증 정보 저장 중 오류 발생: {e}", "error")

    try:
        service = build("gmail", "v1", credentials=creds)
        return service
    except Exception as e:
        log_to_file(f"Gmail 서비스 생성 오류: {e}", "error")
        return None


def send_email(
    recipient_name, recipient_email, subject, html_content, sender_email=None
):
    if not sender_email:
        sender_email = os.getenv("EMAIL_USERNAME")
    if not sender_email:
        log_to_file("발신자 이메일 설정이 되어 있지 않습니다.", "error")
        return False
    try:
        service = get_gmail_service()
        if not service:
            log_to_file("Gmail API 서비스를 초기화할 수 없습니다.", "error")
            return False
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = sender_email
        message["To"] = recipient_email
        message.attach(MIMEText(html_content, "html"))
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        message_body = {"raw": encoded_message}
        sent_message = (
            service.users().messages().send(userId="me", body=message_body).execute()
        )
        log_to_file(
            f"이메일 발송 성공: {recipient_email} (메시지 ID: {sent_message['id']})",
            "info",
        )
        return True
    except Exception as e:
        log_to_file(f"Gmail API를 통한 이메일 발송 오류: {e}", "error")
        try:
            log_to_file("SMTP + OAuth2 방식으로 재시도합니다...", "warning")
            oauth2_token = os.getenv("EMAIL_OAUTH2_TOKEN")
            if not oauth2_token:
                log_to_file("OAuth2 토큰이 설정되지 않았습니다.", "error")
                return False
            smtp_server = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
            smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "587"))
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            auth_string = f"user={sender_email}\1auth=Bearer {oauth2_token}\1\1"
            server.docmd(
                "AUTH", f"XOAUTH2 {base64.b64encode(auth_string.encode()).decode()}"
            )
            server.send_message(message)
            server.quit()
            log_to_file(
                f"SMTP + OAuth2 방식으로 이메일 발송 성공: {recipient_email}", "info"
            )
            return True
        except Exception as smtp_error:
            log_to_file(f"SMTP + OAuth2 방식 이메일 발송 오류: {smtp_error}", "error")
            return False


def generate_api_usage_report():
    try:
        if api_usage_stats["api_calls"] > 0:
            avg_cost_per_call_usd = (
                api_usage_stats["total_cost_usd"] / api_usage_stats["api_calls"]
            )
            avg_cost_per_call_krw = (
                api_usage_stats["total_cost_krw"] / api_usage_stats["api_calls"]
            )
            log_to_file(
                f"\n===== OpenAI API 사용량 통합 보고서 ({datetime.now().strftime('%Y-%m-%d')}) =====\n"
                f"총 API 호출 수: {api_usage_stats['api_calls']}회\n"
                f"총 토큰 사용량: {api_usage_stats['total_tokens']}개\n"
                f" - 입력 토큰: {api_usage_stats['prompt_tokens']}개\n"
                f" - 출력 토큰: {api_usage_stats['completion_tokens']}개\n"
                f"총 비용: ${api_usage_stats['total_cost_usd']:.6f} (약 {api_usage_stats['total_cost_krw']:.2f}원)\n"
                f"평균 호출당 비용: ${avg_cost_per_call_usd:.6f} (약 {avg_cost_per_call_krw:.2f}원)\n",
                "info",
            )
            log_to_file("=== 모델별 사용량 통계 ===", "info")
            for model, stats in api_usage_stats["models"].items():
                model_avg_cost_usd = (
                    stats["total_cost_usd"] / stats["api_calls"]
                    if stats["api_calls"] > 0
                    else 0
                )
                model_avg_cost_krw = (
                    stats["total_cost_krw"] / stats["api_calls"]
                    if stats["api_calls"] > 0
                    else 0
                )
                log_to_file(
                    f"\n모델: {model}\n"
                    f" - API 호출 수: {stats['api_calls']}회\n"
                    f" - 토큰 사용량: {stats['total_tokens']}개 (입력: {stats['prompt_tokens']}개, 출력: {stats['completion_tokens']}개)\n"
                    f" - 총 비용: ${stats['total_cost_usd']:.6f} (약 {stats['total_cost_krw']:.2f}원)\n"
                    f" - 평균 호출당 비용: ${model_avg_cost_usd:.6f} (약 {model_avg_cost_krw:.2f}원)",
                    "info",
                )
            log_to_file("=" * 70, "info")
        else:
            log_to_file(
                "\n===== OpenAI API 사용량 통합 보고서 =====\nAPI 호출 내역이 없습니다.\n"
                + "=" * 70,
                "info",
            )
    except Exception as e:
        log_to_file(f"API 사용량 보고서 생성 중 오류 발생: {e}", "error")


def main():
    try:
        log_to_file("뉴스 크롤링 시작", "info")
        subscribers = load_subscribers()
        log_to_file(f"로드된 구독자 수: {len(subscribers)}", "info")
        if not subscribers:
            log_to_file(
                "구독자가 없습니다. subscribers.txt 파일을 확인하세요.", "error"
            )
            return

        for subscriber in subscribers:
            name = subscriber.get("name", "Unknown")
            email = subscriber.get("email", "")
            topics = subscriber.get("topics", [])
            log_to_file(f"구독자 처리 시작: {name} ({email}), 토픽: {topics}", "info")
            if not topics:
                log_to_file(
                    f"구독자 {name}의 관심 토픽이 없습니다. 건너뜁니다.", "warning"
                )
                continue
            if not email:
                log_to_file(
                    f"구독자 {name}의 이메일 주소가 없습니다. 건너뜁니다.", "warning"
                )
                continue

            log_to_file(f"{name}님의 뉴스 항목 수집 시작", "info")
            subscriber_news_items = fetch_subscriber_news(subscriber)
            if subscriber_news_items:
                log_to_file(
                    f"{name}님을 위한 {len(subscriber_news_items)}개의 뉴스 항목 수집 완료",
                    "info",
                )
                result = process_subscriber(subscriber, subscriber_news_items)
                if result.get("success"):
                    log_to_file(f"{name}님에게 이메일 발송 성공!", "info")
                else:
                    log_to_file(
                        f"{name}님에게 이메일 발송 실패: {result.get('reason', '알 수 없는 오류')}",
                        "error",
                    )
            else:
                log_to_file(f"{name}님의 관심 토픽에 대한 뉴스가 없습니다.", "warning")
            time.sleep(1)

        log_to_file("모든 구독자 처리 완료", "info")
        generate_api_usage_report()
    except Exception as e:
        log_to_file(f"메인 함수 실행 중 오류 발생: {e}", "error")


if __name__ == "__main__":
    main()
