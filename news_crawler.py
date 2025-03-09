#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import xml.etree.ElementTree as ET
import time
from datetime import datetime, timedelta, timezone
import re
import os
from dotenv import load_dotenv
import json
import urllib.parse
import email.utils
import pathlib
import logging
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import sys

# 로깅 설정
logging.basicConfig(
    level=logging.WARNING,  # INFO에서 WARNING으로 변경
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('news_crawler')

# .env 파일 로드
load_dotenv()

# OpenAI API 토큰 사용량 및 비용 추적을 위한 변수
total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens = 0

# GPT-4o-mini 모델 가격 (2024년 기준)
PRICE_PER_1K_PROMPT_TOKENS = 0.00015  # USD per 1K tokens
PRICE_PER_1K_COMPLETION_TOKENS = 0.00060  # USD per 1K tokens
USD_TO_KRW_RATE = 1450  # 달러 대 원화 환율 (변동될 수 있음)

# 사용할 GPT 모델
GPT_MODEL = "gpt-4o-mini"

def get_google_news(query='', country='kr', language='ko'):
    """
    Google News RSS 피드에서 최신 뉴스를 가져오는 함수
    
    매개변수:
        query (str): 검색어 (예: 'ICT기금', '인공지능' 등)
                    기본값은 빈 문자열로, 모든 주제의 뉴스를 가져옵니다.
        country (str): 국가 코드 (예: 'kr', 'us', 'jp' 등)
                      기본값은 'kr'로 한국 뉴스를 가져옵니다.
        language (str): 언어 코드 (예: 'ko', 'en', 'ja' 등)
                       기본값은 'ko'로 한국어 뉴스를 가져옵니다.
    
    반환값:
        list: 뉴스 기사 목록 (각 기사는 딕셔너리 형태)
    """
    # Google News RSS URL 구성
    base_url = "https://news.google.com/rss"
    
    # 검색어가 있는 경우 검색 URL 구성
    if query:
        # URL 인코딩
        encoded_query = urllib.parse.quote(query)
        url = f"{base_url}/search?q={encoded_query}"
    else:
        url = base_url
    
    # 국가 및 언어 파라미터 추가
    url += f"&hl={language}-{country}&gl={country}&ceid={country}:{language}"
    
    logger.info(f"요청 URL: {url}")
    
    try:
        # HTTP 요청 헤더 설정 (User-Agent 추가)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # RSS 피드 요청
        response = requests.get(url, headers=headers, timeout=10)
        
        # 응답 상태 확인
        if response.status_code != 200:
            logger.error(f"오류: HTTP 상태 코드 {response.status_code}")
            return []
        
        # XML 파싱
        root = ET.fromstring(response.content)
        
        # 뉴스 기사 목록 생성
        news_items = []
        
        # 최근 2일 기준 날짜 계산 (시간대 정보 포함)
        two_days_ago = datetime.now(timezone.utc) - timedelta(days=2)
        
        # RSS 채널 내의 아이템(기사) 찾기
        for item in root.findall('.//item'):
            # 기본 정보 추출
            title_elem = item.find('title')
            link_elem = item.find('link')
            pub_date_elem = item.find('pubDate')
            description_elem = item.find('description')
            
            # 게시일 파싱 및 형식 변경
            formatted_date = '날짜 정보 없음'
            is_recent = True  # 기본값을 True로 설정 (날짜 정보가 없는 경우 포함)
            parsed_date = None
            
            if pub_date_elem is not None and pub_date_elem.text:
                try:
                    # RFC 2822 형식의 날짜 파싱 (시간대 정보 포함)
                    parsed_date = email.utils.parsedate_to_datetime(pub_date_elem.text)
                    # yyyy-mm-dd hh:mm 형식으로 변환
                    formatted_date = parsed_date.strftime('%Y-%m-%d %H:%M')
                    
                    # 최근 2일 이내 기사인지 확인
                    is_recent = parsed_date >= two_days_ago
                except Exception as e:
                    logger.warning(f"날짜 파싱 오류: {e}")
                    formatted_date = pub_date_elem.text
                    # 날짜 파싱 오류 시 기본적으로 포함 (True)
                    is_recent = True
            
            # 최근 2일 이내 기사가 아니면 건너뛰기
            if not is_recent:
                continue
            
            # 출처 정보 추출 (여러 방법 시도)
            source = None
            
            # 1. 제목에서 출처 추출 (제목 - 출처 형식)
            if title_elem is not None and title_elem.text:
                title_source_match = re.search(r'^(.*?)\s*-\s*([^-]+)$', title_elem.text)
                if title_source_match:
                    source = title_source_match.group(2).strip()
            
            # 2. description에서 출처 추출
            if source is None and description_elem is not None and description_elem.text:
                # 패턴 1: <font size="-1">출처</font>
                source_match1 = re.search(r'<font size="-1">([^<]+)</font>', description_elem.text)
                if source_match1:
                    source = source_match1.group(1).strip()
                else:
                    # 패턴 2: <b>출처</b>
                    source_match2 = re.search(r'<b>([^<]+)</b>', description_elem.text)
                    if source_match2:
                        source = source_match2.group(1).strip()
            
            # 3. source 태그에서 출처 추출
            if source is None:
                source_elem = item.find('source')
                if source_elem is not None and source_elem.text:
                    source = source_elem.text.strip()
            
            # 기사 정보 구성
            news_item = {
                'title': title_elem.text if title_elem is not None else '제목 없음',
                'link': link_elem.text if link_elem is not None else '#',
                'published': formatted_date,
                'published_raw': pub_date_elem.text if pub_date_elem is not None else None,
                'published_datetime': parsed_date,
                'source': source if source else '알 수 없음',
                'query': query
            }
            
            # 요약 정보 추가
            if description_elem is not None:
                # HTML 태그 제거
                summary = re.sub(r'<[^>]+>', '', description_elem.text)
                news_item['summary'] = summary.strip()
            
            news_items.append(news_item)
        
        logger.info(f"최근 2일 이내 기사 수: {len(news_items)}개")
        
        return news_items
    
    except requests.exceptions.RequestException as e:
        logger.error(f"요청 오류 발생: {e}")
        return []
    except ET.ParseError as e:
        logger.error(f"XML 파싱 오류: {e}")
        return []
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        return []

def get_article_content(url):
    """
    뉴스 기사 URL에서 본문 내용을 가져오는 함수
    BeautifulSoup를 사용하여 더 정확한 내용 추출
    
    매개변수:
        url (str): 뉴스 기사 URL
    
    반환값:
        tuple: (기사 내용, 출처)
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            logger.error(f"오류: HTTP 상태 코드 {response.status_code}")
            return "기사 내용을 가져올 수 없습니다.", None
        
        # BeautifulSoup으로 파싱 (html.parser 사용)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 출처 정보 추출
        source = None
        meta_site_name = soup.find('meta', property='og:site_name')
        if meta_site_name:
            source = meta_site_name.get('content')
        
        # 본문 내용 추출 시도
        content = ""
        
        # 1. article 태그 내용 추출
        article = soup.find('article')
        if article:
            content = article.get_text(strip=True)
        
        # 2. 메타 설명 추출
        if not content:
            meta_desc = soup.find('meta', {'name': 'description'}) or soup.find('meta', property='og:description')
            if meta_desc:
                content = meta_desc.get('content', '')
        
        # 3. 주요 본문 태그에서 내용 추출
        if not content:
            main_content_tags = soup.find_all(['p', 'div'], class_=re.compile(r'article|content|body|text|main', re.I))
            content = ' '.join(tag.get_text(strip=True) for tag in main_content_tags if tag.get_text(strip=True))
        
        # 내용 정제
        if content:
            # 불필요한 공백 제거
            content = re.sub(r'\s+', ' ', content).strip()
            # 광고 문구 등 제거
            content = re.sub(r'(광고|\[광고\]|sponsored content|AD).*?(?=\s|$)', '', content, flags=re.I)
            # 적절한 길이로 자르기
            return content[:2000] + "..." if len(content) > 2000 else content, source
        
        return "기사 내용을 추출할 수 없습니다.", source
    
    except Exception as e:
        logger.error(f"기사 내용 가져오기 오류: {e}")
        return "기사 내용을 가져오는 중 오류가 발생했습니다.", None

def summarize_with_openai(text, title):
    """
    OpenAI API를 사용하여 텍스트를 요약하는 함수
    
    매개변수:
        text (str): 요약할 텍스트
        title (str): 기사 제목
    
    반환값:
        str: 요약된 텍스트
    """
    global total_prompt_tokens, total_completion_tokens, total_tokens
    
    # OpenAI API 키 가져오기
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key or api_key == 'your_openai_api_key_here':
        logger.error("OpenAI API 키가 설정되지 않았습니다.")
        return "OpenAI API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 설정해주세요."
    
    try:
        # OpenAI API 엔드포인트
        url = "https://api.openai.com/v1/chat/completions"
        
        # 요청 헤더
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 개선된 요약 프롬프트
        system_prompt = """당신은 뉴스 기사를 명확하고 간결하게 요약하는 전문가입니다. 
다음 규칙을 따라 요약해주세요:
1. 기사의 핵심 내용과 중요 사실만 포함하세요.
2. 3-4개의 간결한 문장으로 요약하세요.
3. 원문의 중요한 정보를 누락하지 마세요.
4. 객관적인 사실만 포함하고 의견이나 해석은 배제하세요.
5. 요약은 기사의 전체 맥락을 이해할 수 있도록 작성하세요."""
        
        # 요청 데이터
        data = {
            "model": GPT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"제목: {title}\n\n내용: {text}\n\n위 뉴스 기사를 요약해주세요."
                }
            ],
            "temperature": 0.3,  # 더 일관된 결과를 위해 온도 낮춤
            "max_tokens": 300
        }
        
        # API 요청
        logger.info(f"OpenAI API 요청 중: {GPT_MODEL} 모델 사용")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        # 응답 확인
        if response.status_code == 200:
            result = response.json()
            summary = result['choices'][0]['message']['content'].strip()
            
            # 토큰 사용량 추적
            if 'usage' in result:
                prompt_tokens = result['usage'].get('prompt_tokens', 0)
                completion_tokens = result['usage'].get('completion_tokens', 0)
                tokens = result['usage'].get('total_tokens', 0)
                
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_tokens += tokens
                
                logger.info(f"API 호출 토큰 사용량: 프롬프트 {prompt_tokens}, 완성 {completion_tokens}, 총 {tokens}")
            
            return summary
        else:
            logger.error(f"OpenAI API 오류: {response.status_code}")
            error_detail = response.json() if response.content else "상세 정보 없음"
            logger.error(f"오류 상세: {error_detail}")
            return f"요약 생성 중 오류가 발생했습니다. 상태 코드: {response.status_code}"
    
    except Exception as e:
        logger.error(f"OpenAI API 요청 오류: {e}")
        return "요약 생성 중 오류가 발생했습니다."

def display_news(news_items, limit=10):
    """
    뉴스 기사 목록을 콘솔에 출력하는 함수
    
    매개변수:
        news_items (list): 뉴스 기사 목록
        limit (int): 출력할 기사 수 (기본값: 10)
    """
    print(f"\n{'=' * 80}")
    print(f"{'최근 2일 이내 뉴스 목록':^80}")
    print(f"{'=' * 80}")
    
    # 기사 수 제한
    items_to_display = news_items[:limit]
    
    for i, item in enumerate(items_to_display, 1):
        print(f"\n[{i}] {item['title']}")
        print(f"검색어: {item['query'] if item['query'] else '일반'}")
        print(f"출처: {item['source']}")
        print(f"게시일: {item['published']}")
        print(f"링크: {item['link']}")
        if 'ai_summary' in item:
            print(f"AI 요약: {item['ai_summary']}")
        elif 'summary' in item:
            print(f"요약: {item['summary']}")
        print(f"{'-' * 80}")

def save_to_file(news_items, result_dir="result", include_token_info=True, no_result_topics=None):
    """
    뉴스 기사 목록을 파일에 저장하는 함수
    
    매개변수:
        news_items (list): 뉴스 기사 목록
        result_dir (str): 결과 파일을 저장할 디렉토리 (기본값: "result")
        include_token_info (bool): 토큰 사용량 정보 포함 여부
        no_result_topics (list): 검색 결과가 없는 토픽 목록
    
    반환값:
        str: 저장된 파일 경로
    """
    try:
        # 결과 디렉토리 생성 (없는 경우)
        pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)
        
        # 현재 시간을 파일명에 포함
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{result_dir}/result_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"{'최근 2일 이내 뉴스 목록':^80}\n")
            f.write(f"{'=' * 80}\n\n")
            
            for i, item in enumerate(news_items, 1):
                f.write(f"[{i}] {item['title']}\n")
                f.write(f"검색어: {item['query'] if item['query'] else '일반'}\n")
                f.write(f"출처: {item['source']}\n")
                f.write(f"게시일: {item['published']}\n")
                f.write(f"링크: {item['link']}\n")
                
                if 'ai_summary' in item:
                    f.write(f"AI 요약: {item['ai_summary']}\n")
                elif 'summary' in item:
                    f.write(f"요약: {item['summary']}\n")
                
                f.write(f"{'-' * 80}\n\n")
            
            f.write(f"\n저장 시간: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"검색 기간: 최근 2일 이내 ({(datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')} ~ {datetime.now().strftime('%Y-%m-%d')})\n")
            
            # 검색 결과가 없는 토픽 정보 추가
            if no_result_topics:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"{'검색 결과가 없는 토픽':^80}\n")
                f.write(f"{'=' * 80}\n\n")
                for topic in no_result_topics:
                    f.write(f"- {topic}\n")
                f.write("\n")
            
            # 토큰 사용량 정보 추가
            if include_token_info and total_tokens > 0:
                cost_info = calculate_token_cost()
                
                f.write(f"\n{'=' * 80}\n")
                f.write(f"{'OpenAI API 토큰 사용량 정보':^80}\n")
                f.write(f"{'=' * 80}\n\n")
                f.write(f"사용 모델: {GPT_MODEL}\n")
                f.write(f"프롬프트 토큰: {cost_info['prompt_tokens']:,}개\n")
                f.write(f"완성 토큰: {cost_info['completion_tokens']:,}개\n")
                f.write(f"총 토큰: {cost_info['total_tokens']:,}개\n\n")
                f.write(f"예상 비용: ${cost_info['total_cost_usd']:.6f} (약 {cost_info['total_cost_krw']:.2f}원)\n")
                f.write(f"참고: https://platform.openai.com/settings/organization/limits\n")
        
        logger.info(f"뉴스 기사가 '{filename}' 파일에 저장되었습니다.")
        return filename
    
    except Exception as e:
        logger.error(f"파일 저장 오류: {e}")
        return None

def calculate_token_cost():
    """
    OpenAI API 토큰 사용량 및 비용을 계산하는 함수
    
    반환값:
        dict: 토큰 사용량 및 비용 정보
    """
    prompt_cost = (total_prompt_tokens / 1000) * PRICE_PER_1K_PROMPT_TOKENS
    completion_cost = (total_completion_tokens / 1000) * PRICE_PER_1K_COMPLETION_TOKENS
    total_cost = prompt_cost + completion_cost
    total_cost_krw = total_cost * USD_TO_KRW_RATE
    
    return {
        'prompt_tokens': total_prompt_tokens,
        'completion_tokens': total_completion_tokens,
        'total_tokens': total_tokens,
        'prompt_cost_usd': prompt_cost,
        'completion_cost_usd': completion_cost,
        'total_cost_usd': total_cost,
        'total_cost_krw': total_cost_krw
    }

def display_token_info():
    """
    OpenAI API 토큰 사용량 및 비용 정보를 콘솔에 출력하는 함수
    """
    if total_tokens == 0:
        logger.info("OpenAI API를 사용하지 않았거나 토큰 정보를 가져오지 못했습니다.")
        return
    
    cost_info = calculate_token_cost()
    
    print(f"\n{'=' * 80}")
    print(f"{'OpenAI API 토큰 사용량 정보':^80}")
    print(f"{'=' * 80}")
    print(f"사용 모델: {GPT_MODEL}")
    print(f"프롬프트 토큰: {cost_info['prompt_tokens']:,}개")
    print(f"완성 토큰: {cost_info['completion_tokens']:,}개")
    print(f"총 토큰: {cost_info['total_tokens']:,}개")
    print(f"\n예상 비용: ${cost_info['total_cost_usd']:.6f} (약 {cost_info['total_cost_krw']:.2f}원)")
    print(f"참고: https://platform.openai.com/settings/organization/limits")
    print(f"{'=' * 80}")

def process_news_item(item):
    """
    개별 뉴스 아이템을 처리하는 함수
    
    매개변수:
        item (dict): 처리할 뉴스 아이템
    
    반환값:
        dict: 처리된 뉴스 아이템
    """
    try:
        # 기사 내용 가져오기
        article_content, article_source = get_article_content(item['link'])
        
        # 기사 출처 업데이트
        if article_source and item['source'] == '알 수 없음':
            item['source'] = article_source
        
        # OpenAI API로 요약하기
        ai_summary = summarize_with_openai(article_content, item['title'])
        item['ai_summary'] = ai_summary
        
        return item
    except Exception as e:
        logger.error(f"뉴스 아이템 처리 오류: {e}")
        return item

def main():
    """
    메인 함수 - 프로그램 실행 시작점
    """
    print("\nGoogle News RSS 피드에서 최근 2일 이내 뉴스 가져오기")
    print(f"검색 기간: {(datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')} ~ {datetime.now().strftime('%Y-%m-%d')}")
    
    # .env 파일에서 검색어 가져오기
    queries_str = os.getenv('NEWS_TOPICS', 'ICT기금')
    queries = [query.strip() for query in queries_str.split(',')]
    
    all_news_items = []
    total_articles = 0
    no_result_topics = []  # 검색 결과가 없는 토픽을 저장할 리스트
    
    # 전체 진행률을 표시하는 tqdm 설정
    with tqdm(
        total=len(queries),
        desc="전체 진행률",
        unit="주제",
        dynamic_ncols=False,  # 동적 크기 조절 비활성화
        ncols=100,  # 진행률 바의 전체 너비 고정
        position=0,
        leave=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'  # 진행률 바 포맷 지정
    ) as main_pbar:
        # 각 검색어별로 뉴스 가져오기
        for query in queries:
            query_news = get_google_news(query=query)
            
            if query_news:
                # 각 검색어별로 최대 5개의 뉴스만 선택
                selected_news = query_news[:5]
                total_articles += len(selected_news)
                
                # 병렬 처리로 뉴스 아이템 처리
                with ThreadPoolExecutor(max_workers=3) as executor:
                    processed_items = []
                    futures = {executor.submit(process_news_item, item): item for item in selected_news}
                    
                    for future in futures:
                        try:
                            result = future.result()
                            processed_items.append(result)
                        except Exception as e:
                            logger.error(f"뉴스 처리 중 오류 발생: {e}")
                
                all_news_items.extend(processed_items)
            else:
                no_result_topics.append(query)  # 검색 결과가 없는 토픽 추가
            
            main_pbar.update(1)
    
    # 모든 뉴스 출력 및 저장
    if all_news_items:
        print(f"\n총 {len(all_news_items)}개의 뉴스 기사 처리 완료")
        
        # 결과를 JSON 형식으로도 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_filename = f"result/result_{timestamp}.json"
        
        # JSON 파일 저장
        try:
            pathlib.Path("result").mkdir(parents=True, exist_ok=True)
            with open(json_filename, 'w', encoding='utf-8') as f:
                # datetime 객체를 문자열로 변환
                json_data = []
                for item in all_news_items:
                    item_copy = item.copy()
                    if 'published_datetime' in item_copy:
                        item_copy['published_datetime'] = item_copy['published_datetime'].isoformat() if item_copy['published_datetime'] else None
                    json_data.append(item_copy)
                
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"결과가 '{json_filename}' 파일에 저장되었습니다.")
        except Exception as e:
            logger.error(f"JSON 파일 저장 오류: {e}")
        
        # 기존 텍스트 파일 형식으로도 저장
        saved_file = save_to_file(all_news_items, no_result_topics=no_result_topics)
        if saved_file:
            print(f"텍스트 결과가 '{saved_file}' 파일에 저장되었습니다.")
            if no_result_topics:
                print("\n검색 결과가 없는 토픽:")
                for topic in no_result_topics:
                    print(f"- {topic}")
        
        # 토큰 사용량 및 비용 정보 출력
        display_token_info()
    else:
        print("\n최근 2일 이내 뉴스를 가져오지 못했습니다.")
        if no_result_topics:
            print("\n검색 결과가 없는 토픽:")
            for topic in no_result_topics:
                print(f"- {topic}")

if __name__ == "__main__":
    main() 