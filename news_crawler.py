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

# .env 파일 로드
load_dotenv()

# OpenAI API 토큰 사용량 및 비용 추적을 위한 변수
total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens = 0

# GPT-3.5-turbo 모델 가격 (2023년 기준, 변경될 수 있음)
PRICE_PER_1K_PROMPT_TOKENS = 0.0015  # USD per 1K tokens
PRICE_PER_1K_COMPLETION_TOKENS = 0.002  # USD per 1K tokens
USD_TO_KRW_RATE = 1350  # 달러 대 원화 환율 (변동될 수 있음)

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
    
    print(f"요청 URL: {url}")
    
    try:
        # HTTP 요청 헤더 설정 (User-Agent 추가)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # RSS 피드 요청
        response = requests.get(url, headers=headers, timeout=10)
        
        # 응답 상태 확인
        if response.status_code != 200:
            print(f"오류: HTTP 상태 코드 {response.status_code}")
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
                    print(f"날짜 파싱 오류: {e}")
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
        
        print(f"최근 2일 이내 기사 수: {len(news_items)}개")
        
        return news_items
    
    except requests.exceptions.RequestException as e:
        print(f"요청 오류 발생: {e}")
        return []
    except ET.ParseError as e:
        print(f"XML 파싱 오류: {e}")
        return []
    except Exception as e:
        print(f"오류 발생: {e}")
        return []

def get_article_content(url):
    """
    뉴스 기사 URL에서 본문 내용을 가져오는 함수
    
    매개변수:
        url (str): 뉴스 기사 URL
    
    반환값:
        str: 뉴스 기사 본문 내용
    """
    try:
        # HTTP 요청 헤더 설정 (User-Agent 추가)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # 뉴스 기사 요청
        response = requests.get(url, headers=headers, timeout=10)
        
        # 응답 상태 확인
        if response.status_code != 200:
            print(f"오류: HTTP 상태 코드 {response.status_code}")
            return "기사 내용을 가져올 수 없습니다.", None
        
        # 간단한 본문 추출 (메타 태그 또는 본문 태그에서 추출)
        # 실제로는 각 뉴스 사이트마다 구조가 다르므로 더 복잡한 파싱이 필요할 수 있음
        content = response.text
        
        # 출처 정보 추출 시도
        source = None
        
        # 메타 태그에서 출처 추출
        source_match = re.search(r'<meta[^>]*property=["\']og:site_name["\'][^>]*content=["\'](.*?)["\']', content, re.IGNORECASE)
        if source_match:
            source = source_match.group(1).strip()
        
        # 메타 태그에서 설명 추출
        description_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\'](.*?)["\']', content, re.IGNORECASE)
        if description_match:
            return description_match.group(1), source
        
        # 본문 태그에서 텍스트 추출 (간단한 방식)
        body_match = re.search(r'<body[^>]*>(.*?)</body>', content, re.IGNORECASE | re.DOTALL)
        if body_match:
            # HTML 태그 제거
            body_text = re.sub(r'<[^>]+>', ' ', body_match.group(1))
            # 여러 공백 제거
            body_text = re.sub(r'\s+', ' ', body_text).strip()
            # 적절한 길이로 자르기
            return body_text[:1000] + "..." if len(body_text) > 1000 else body_text, source
        
        return "기사 내용을 추출할 수 없습니다.", source
    
    except Exception as e:
        print(f"기사 내용 가져오기 오류: {e}")
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
        return "OpenAI API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 설정해주세요."
    
    try:
        # OpenAI API 엔드포인트
        url = "https://api.openai.com/v1/chat/completions"
        
        # 요청 헤더
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 요청 데이터
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": "당신은 뉴스 기사를 간결하게 요약하는 전문가입니다. 주어진 뉴스 기사를 3-4문장으로 요약해주세요."
                },
                {
                    "role": "user",
                    "content": f"제목: {title}\n\n내용: {text}\n\n이 뉴스 기사를 3-4문장으로 요약해주세요."
                }
            ],
            "temperature": 0.5,
            "max_tokens": 300
        }
        
        # API 요청
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
                
                print(f"API 호출 토큰 사용량: 프롬프트 {prompt_tokens}, 완성 {completion_tokens}, 총 {tokens}")
            
            return summary
        else:
            print(f"OpenAI API 오류: {response.status_code}")
            return f"요약 생성 중 오류가 발생했습니다. 상태 코드: {response.status_code}"
    
    except Exception as e:
        print(f"OpenAI API 요청 오류: {e}")
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

def save_to_file(news_items, result_dir="result", include_token_info=True):
    """
    뉴스 기사 목록을 파일에 저장하는 함수
    
    매개변수:
        news_items (list): 뉴스 기사 목록
        result_dir (str): 결과 파일을 저장할 디렉토리 (기본값: "result")
        include_token_info (bool): 토큰 사용량 정보 포함 여부
    
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
            
            # 토큰 사용량 정보 추가
            if include_token_info and total_tokens > 0:
                prompt_cost = (total_prompt_tokens / 1000) * PRICE_PER_1K_PROMPT_TOKENS
                completion_cost = (total_completion_tokens / 1000) * PRICE_PER_1K_COMPLETION_TOKENS
                total_cost = prompt_cost + completion_cost
                total_cost_krw = total_cost * USD_TO_KRW_RATE
                
                f.write(f"\n{'=' * 80}\n")
                f.write(f"{'OpenAI API 토큰 사용량 정보':^80}\n")
                f.write(f"{'=' * 80}\n\n")
                f.write(f"프롬프트 토큰: {total_prompt_tokens:,}개\n")
                f.write(f"완성 토큰: {total_completion_tokens:,}개\n")
                f.write(f"총 토큰: {total_tokens:,}개\n\n")
                f.write(f"예상 비용: ${total_cost:.6f} (약 {total_cost_krw:.2f}원)\n")
                f.write(f"참고: https://platform.openai.com/settings/organization/limits\n")
        
        print(f"\n뉴스 기사가 '{filename}' 파일에 저장되었습니다.")
        return filename
    
    except Exception as e:
        print(f"파일 저장 오류: {e}")
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
        print("\nOpenAI API를 사용하지 않았거나 토큰 정보를 가져오지 못했습니다.")
        return
    
    cost_info = calculate_token_cost()
    
    print(f"\n{'=' * 80}")
    print(f"{'OpenAI API 토큰 사용량 정보':^80}")
    print(f"{'=' * 80}")
    print(f"프롬프트 토큰: {cost_info['prompt_tokens']:,}개")
    print(f"완성 토큰: {cost_info['completion_tokens']:,}개")
    print(f"총 토큰: {cost_info['total_tokens']:,}개")
    print(f"\n예상 비용: ${cost_info['total_cost_usd']:.6f} (약 {cost_info['total_cost_krw']:.2f}원)")
    print(f"참고: https://platform.openai.com/settings/organization/limits")
    print(f"{'=' * 80}")

def main():
    """
    메인 함수 - 프로그램 실행 시작점
    """
    print("Google News RSS 피드에서 최근 2일 이내 뉴스 가져오기")
    print(f"검색 기간: {(datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')} ~ {datetime.now().strftime('%Y-%m-%d')}")
    
    # .env 파일에서 검색어 가져오기
    queries_str = os.getenv('NEWS_TOPICS', 'ICT기금')
    queries = [query.strip() for query in queries_str.split(',')]
    
    all_news_items = []
    
    # 각 검색어별로 뉴스 가져오기
    for query in queries:
        print(f"\n'{query}' 관련 뉴스 가져오는 중...")
        query_news = get_google_news(query=query)
        
        if query_news:
            print(f"'{query}' 관련 최근 2일 이내 뉴스 {len(query_news)}개를 찾았습니다.")
            
            # 각 검색어별로 최대 3개의 뉴스만 처리
            for item in query_news[:3]:
                print(f"'{item['title']}' 기사 처리 중...")
                
                # 기사 내용 가져오기
                article_content, article_source = get_article_content(item['link'])
                
                # 기사 출처 업데이트 (기사 내용에서 추출한 출처가 있는 경우)
                if article_source and item['source'] == '알 수 없음':
                    item['source'] = article_source
                
                # OpenAI API로 요약하기
                ai_summary = summarize_with_openai(article_content, item['title'])
                item['ai_summary'] = ai_summary
                
                all_news_items.append(item)
        else:
            print(f"'{query}' 관련 최근 2일 이내 뉴스를 가져오지 못했습니다.")
    
    # 모든 뉴스 출력
    if all_news_items:
        print(f"\n총 {len(all_news_items)}개의 최근 2일 이내 뉴스 기사를 처리했습니다.")
        display_news(all_news_items)
        
        # 결과를 파일에 저장
        save_to_file(all_news_items)
        
        # 토큰 사용량 및 비용 정보 출력
        display_token_info()
    else:
        print("최근 2일 이내 뉴스를 가져오지 못했습니다.")

if __name__ == "__main__":
    main() 