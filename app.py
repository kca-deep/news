#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import pathlib
import bcrypt
import secrets
import functools
import datetime
import uuid
import re
import binascii
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify,
    abort,
    get_flashed_messages,
)
from flask_wtf.csrf import CSRFProtect
from dotenv import load_dotenv
from datetime import timedelta

# .env 파일 로드
load_dotenv()

# --- 로깅 설정 ---
pathlib.Path("logs").mkdir(parents=True, exist_ok=True)
log_file = "logs/app.log"
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG, INFO, WARNING, ERROR, CRITICAL 모두 기록
    handlers=[
        logging.FileHandler(log_file, mode="a", encoding="utf-8", errors="replace"),
    ],
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def log_to_file(message):
    """중요 메시지(경고 이상)만 기록"""
    logging.warning(message)


# --- 파일 입출력 헬퍼 함수 ---
def write_subscribers(subscribers, file_path="subscribers.txt"):
    """구독자 목록을 지정 파일에 저장하는 함수."""
    try:
        with open(file_path, "w", encoding="utf-8", errors="replace") as f:
            for i, subscriber in enumerate(subscribers):
                if i > 0:
                    f.write("\n")
                f.write(f"ID: {subscriber['id']}\n")
                f.write(f"이름: {subscriber['name']}\n")
                f.write(f"이메일: {subscriber['email']}\n")

                # 토픽 리스트를 쉼표로 구분된 문자열로 변환하여 저장
                topics = subscriber.get("topics", [])
                if isinstance(topics, list):
                    topics_str = ", ".join(topics)
                else:
                    # 이미 문자열인 경우 그대로 사용
                    topics_str = str(topics)

                f.write(f"토픽: {topics_str}\n")
    except Exception as e:
        logging.error(f"구독자 파일 쓰기 오류: {e}")


# --- 구독자 관리 함수 ---
def load_subscribers(file_path="subscribers.txt"):
    """
    텍스트 파일에서 구독자 정보를 로드하는 함수.
    구독자 정보가 불완전한 경우 UUID를 생성하고, 파일 정리가 필요하면 파일을 다시 씁니다.
    """
    subscribers = []
    file_needs_update = False
    needs_cleanup = False

    if not os.path.exists(file_path):
        return subscribers

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            current_subscriber = {}
            for line in f:
                line = line.strip()
                if not line:
                    # 빈 줄을 만난 경우, 하나의 구독자 정보 완성
                    if current_subscriber.get("name") and current_subscriber.get(
                        "email"
                    ):
                        if not current_subscriber.get("id"):
                            current_subscriber["id"] = str(uuid.uuid4())
                            file_needs_update = True

                        # 토픽 문자열을 일관된 리스트 형식으로 변환
                        topics = current_subscriber.get("topics", "")
                        # 문자열이 리스트 형식으로 저장되어 있는 경우 처리 (예: "['AI', 'MCP']")
                        if isinstance(topics, str):
                            if topics.startswith("[") and topics.endswith("]"):
                                try:
                                    # 문자열로 된 리스트를 실제 리스트로 변환
                                    topics = eval(topics)
                                except:
                                    # 변환 실패 시 쉼표로 구분
                                    topics = [
                                        t.strip()
                                        for t in topics.replace("[", "")
                                        .replace("]", "")
                                        .replace("'", "")
                                        .split(",")
                                        if t.strip()
                                    ]
                            else:
                                # 일반 문자열은 쉼표로 구분하여 리스트로 변환
                                topics = [
                                    t.strip() for t in topics.split(",") if t.strip()
                                ]

                        # 빈 리스트가 아니면 업데이트
                        if topics:
                            current_subscriber["topics"] = topics
                            file_needs_update = True

                        subscribers.append(current_subscriber)
                    else:
                        needs_cleanup = True
                    current_subscriber = {}
                    continue

                if line.startswith("ID:"):
                    # 이전 구독자 정보가 있다면 저장
                    if current_subscriber:
                        if current_subscriber.get("name") and current_subscriber.get(
                            "email"
                        ):
                            if not current_subscriber.get("id"):
                                current_subscriber["id"] = str(uuid.uuid4())
                                file_needs_update = True

                            # 토픽 문자열을 일관된 리스트 형식으로 변환
                            topics = current_subscriber.get("topics", "")
                            # 문자열이 리스트 형식으로 저장되어 있는 경우 처리 (예: "['AI', 'MCP']")
                            if isinstance(topics, str):
                                if topics.startswith("[") and topics.endswith("]"):
                                    try:
                                        # 문자열로 된 리스트를 실제 리스트로 변환
                                        topics = eval(topics)
                                    except:
                                        # 변환 실패 시 쉼표로 구분
                                        topics = [
                                            t.strip()
                                            for t in topics.replace("[", "")
                                            .replace("]", "")
                                            .replace("'", "")
                                            .split(",")
                                            if t.strip()
                                        ]
                                else:
                                    # 일반 문자열은 쉼표로 구분하여 리스트로 변환
                                    topics = [
                                        t.strip()
                                        for t in topics.split(",")
                                        if t.strip()
                                    ]

                            # 빈 리스트가 아니면 업데이트
                            if topics:
                                current_subscriber["topics"] = topics
                                file_needs_update = True

                            subscribers.append(current_subscriber)
                        else:
                            needs_cleanup = True
                    current_subscriber = {
                        "id": line[3:].strip(),
                        "name": "",
                        "email": "",
                        "topics": [],
                    }
                elif line.startswith("이름:"):
                    current_subscriber["name"] = line[3:].strip()
                elif line.startswith("이메일:"):
                    current_subscriber["email"] = line[4:].strip()
                elif line.startswith("토픽:"):
                    topics_str = line[3:].strip()
                    # 문자열이 리스트 형식으로 저장되어 있는 경우 처리 (예: "['AI', 'MCP']")
                    if topics_str.startswith("[") and topics_str.endswith("]"):
                        try:
                            # 문자열로 된 리스트를 실제 리스트로 변환
                            topics = eval(topics_str)
                        except:
                            # 변환 실패 시 쉼표로 구분
                            topics = [
                                t.strip()
                                for t in topics_str.replace("[", "")
                                .replace("]", "")
                                .replace("'", "")
                                .split(",")
                                if t.strip()
                            ]
                    else:
                        # 일반 문자열은 쉼표로 구분하여 리스트로 변환
                        topics = [t.strip() for t in topics_str.split(",") if t.strip()]

                    current_subscriber["topics"] = topics

            # 마지막 구독자 정보 처리
            if current_subscriber:
                if current_subscriber.get("name") and current_subscriber.get("email"):
                    if not current_subscriber.get("id"):
                        current_subscriber["id"] = str(uuid.uuid4())
                        file_needs_update = True

                    # 토픽 문자열을 일관된 리스트 형식으로 변환
                    topics = current_subscriber.get("topics", "")
                    # 문자열이 리스트 형식으로 저장되어 있는 경우 처리 (예: "['AI', 'MCP']")
                    if isinstance(topics, str):
                        if topics.startswith("[") and topics.endswith("]"):
                            try:
                                # 문자열로 된 리스트를 실제 리스트로 변환
                                topics = eval(topics)
                            except:
                                # 변환 실패 시 쉼표로 구분
                                topics = [
                                    t.strip()
                                    for t in topics.replace("[", "")
                                    .replace("]", "")
                                    .replace("'", "")
                                    .split(",")
                                    if t.strip()
                                ]
                        else:
                            # 일반 문자열은 쉼표로 구분하여 리스트로 변환
                            topics = [t.strip() for t in topics.split(",") if t.strip()]

                    # 빈 리스트가 아니면 업데이트
                    if topics:
                        current_subscriber["topics"] = topics
                        file_needs_update = True

                    subscribers.append(current_subscriber)
                else:
                    needs_cleanup = True

        # 파일 정리나 업데이트가 필요하면 다시 씁니다.
        if (needs_cleanup or file_needs_update) and subscribers:
            write_subscribers(subscribers, file_path)

        return subscribers

    except Exception as e:
        log_to_file(f"구독자 정보 로드 오류: {e}")
        return []


def save_subscriber(name, email, topics, file_path="subscribers.txt"):
    """새 구독자 정보를 파일에 저장하는 함수."""
    try:
        subscribers = load_subscribers(file_path)
        # 이미 등록된 이메일이 있는지 확인
        for subscriber in subscribers:
            if subscriber["email"] == email:
                logging.info(f"구독자 추가 실패: 이미 등록된 이메일 - {email}")
                return False

        # 새 구독자 추가 (UUID 생성)
        new_subscriber = {
            "id": str(uuid.uuid4()),
            "name": name,
            "email": email,
            "topics": topics,
        }
        subscribers.append(new_subscriber)
        write_subscribers(subscribers, file_path)
        logging.info(f"구독자 추가: {name} ({email})")
        return True

    except Exception as e:
        logging.error(f"구독자 추가 오류: {e}")
        return False


def delete_subscriber_by_id(subscriber_id, file_path="subscribers.txt"):
    """파일에서 구독자 정보를 ID로 삭제하는 함수."""
    try:
        subscribers = load_subscribers(file_path)
        new_subscribers = [sub for sub in subscribers if sub["id"] != subscriber_id]
        if len(new_subscribers) == len(subscribers):
            logging.info(f"구독자 삭제 실패: ID {subscriber_id}를 찾을 수 없습니다.")
            return False
        write_subscribers(new_subscribers, file_path)
        logging.info(f"구독자 삭제: ID {subscriber_id}")
        return True

    except Exception as e:
        logging.error(f"구독자 삭제 오류: {e}")
        return False


def update_subscriber_by_id(
    subscriber_id, name, email, topics, file_path="subscribers.txt"
):
    """파일에서 구독자 정보를 ID로 찾아 수정하는 함수."""
    try:
        subscribers = load_subscribers(file_path)
        updated = False

        for subscriber in subscribers:
            if subscriber["id"] == subscriber_id:
                # 이메일 변경 시 중복 검사
                if subscriber["email"] != email:
                    if any(
                        other["email"] == email
                        for other in subscribers
                        if other["id"] != subscriber_id
                    ):
                        return False, "이미 존재하는 이메일입니다."
                subscriber["name"] = name
                subscriber["email"] = email
                subscriber["topics"] = topics
                updated = True
                break

        if not updated:
            return False, "구독자를 찾을 수 없습니다."

        write_subscribers(subscribers, file_path)
        logging.info(
            f"구독자 정보 수정: ID {subscriber_id}, 이름: {name}, 이메일: {email}"
        )
        return True, ""
    except Exception as e:
        logging.error(f"구독자 정보 수정 오류: {e}")
        return False, f"수정 중 오류 발생: {e}"


def is_valid_email(email):
    """이메일 형식이 유효한지 확인하는 함수."""
    email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(email_regex, email) is not None


# --- 비밀번호 해싱 관련 함수 ---
def hash_password(password):
    """비밀번호를 bcrypt로 해싱하는 함수."""
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def check_password(plain_password, hashed_password):
    """해시된 비밀번호와 일반 텍스트 비밀번호가 일치하는지 확인하는 함수."""
    plain_bytes = plain_password.encode("utf-8")
    hashed_bytes = hashed_password.encode("utf-8")
    return bcrypt.checkpw(plain_bytes, hashed_bytes)


# --- 세션 필요 데코레이터 ---
def login_required(func):
    """로그인이 필요한 라우트에 적용할 데코레이터."""

    @functools.wraps(func)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            flash("로그인이 필요한 페이지입니다.")
            return redirect(url_for("login"))
        return func(*args, **kwargs)

    return decorated_function


# Flask 웹앱 설정
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "kca-newsletter-secret-key")


# CSRF 보호
@app.before_request
def csrf_protect():
    if request.method == "POST":
        # 로그인 페이지에 대한 POST 요청은 CSRF 검증을 건너뜁니다.
        if request.path == "/login" or request.path == "/":
            return
        token = session.get("csrf_token")
        if not token or token != request.form.get("csrf_token"):
            abort(403)


@app.before_request
def make_session_permanent():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(hours=1)


# CSRF 토큰 생성
@app.context_processor
def inject_csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = binascii.hexlify(os.urandom(16)).decode()
    return dict(csrf_token=session["csrf_token"])


# 웹 라우트: 로그인
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    # POST 요청 처리 (로그인 시도)
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == "kca" and password == "admin123":
            session["logged_in"] = True
            session["username"] = username
            session["login_time"] = datetime.datetime.now().isoformat()
            log_to_file(f"관리자 로그인 성공: {username}")
            return redirect(url_for("list_subscribers"))
        else:
            flash(
                "로그인에 실패했습니다. 사용자 이름과 비밀번호를 확인하세요.", "error"
            )
            log_to_file(f"로그인 실패 시도: {username}")

    # 이미 로그인되어 있으면 구독자 목록으로 리디렉션
    if "logged_in" in session:
        return redirect(url_for("list_subscribers"))

    # GET 요청 시 로그인 페이지 표시
    return render_template("login.html")


# 웹 라우트: 로그아웃
@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    flash("로그아웃되었습니다.", "success")
    return redirect(url_for("login"))


@app.route("/subscribers", methods=["GET"])
def list_subscribers():
    # 로그인 확인
    if "logged_in" not in session:
        return redirect(url_for("login"))

    # 검색어와 토픽 필터 처리
    search_query = request.args.get("search", "").strip().lower()
    topic_filter = request.args.get("topic", "").strip().lower()

    # 구독자 목록 가져오기
    subscribers = load_subscribers()

    # 모든 토픽 추출
    all_topics = set()
    for subscriber in subscribers:
        all_topics.update(subscriber.get("topics", []))
    all_topics = sorted(list(all_topics))

    # 검색어에 따른 필터링
    if search_query:
        filtered_subscribers = []
        for subscriber in subscribers:
            if (
                search_query in subscriber["name"].lower()
                or search_query in subscriber["email"].lower()
            ):
                filtered_subscribers.append(subscriber)
        subscribers = filtered_subscribers

    # 토픽에 따른 필터링
    if topic_filter:
        filtered_subscribers = []
        for subscriber in subscribers:
            if topic_filter in [t.lower() for t in subscriber.get("topics", [])]:
                filtered_subscribers.append(subscriber)
        subscribers = filtered_subscribers

    log_to_file(f"구독자 목록 조회: {len(subscribers)}명")
    return render_template(
        "list_subscribers.html",
        subscribers=subscribers,
        all_topics=all_topics,
        request=request,
    )


@app.route("/subscribers/add", methods=["GET", "POST"])
def add_subscriber():
    # 로그인 확인
    if "logged_in" not in session:
        return redirect(url_for("login"))

    # CSRF 보호
    csrf = request.form.get("csrf_token")
    if request.method == "POST" and csrf and csrf == session.get("csrf_token"):
        # 폼 데이터 가져오기
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        topics_str = request.form.get("topics", "").strip()

        # 토픽을 쉼표로 분리하여 리스트로 변환
        topics = [topic.strip() for topic in topics_str.split(",") if topic.strip()]

        # 이메일 형식 확인
        if not is_valid_email(email):
            flash("유효하지 않은 이메일 형식입니다.", "error")
            return render_template("subscriber_form.html", mode="add")

        # 구독자 ID 생성
        subscriber_id = str(uuid.uuid4())

        # 새 구독자 정보 생성
        new_subscriber = {
            "id": subscriber_id,
            "name": name,
            "email": email,
            "topics": topics,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 구독자 목록에 추가
        subscribers = load_subscribers()
        subscribers.append(new_subscriber)
        write_subscribers(subscribers)

        # 로그 기록
        log_to_file(f"구독자 추가: {name} ({email})")

        flash("구독자가 성공적으로 추가되었습니다.", "success")
        return redirect(url_for("list_subscribers"))

    # GET 요청 시 폼 표시
    return render_template("subscriber_form.html", mode="add")


@app.route("/subscribers/edit/<subscriber_id>", methods=["GET", "POST"])
def edit_subscriber(subscriber_id):
    # 로그인 확인
    if "logged_in" not in session:
        return redirect(url_for("login"))

    # 구독자 정보 가져오기
    subscribers = load_subscribers()
    subscriber = None
    for s in subscribers:
        if s.get("id") == subscriber_id:
            subscriber = s
            break

    # 구독자가 존재하지 않는 경우
    if not subscriber:
        flash("구독자를 찾을 수 없습니다.", "error")
        return redirect(url_for("list_subscribers"))

    # POST 요청 처리 (구독자 정보 업데이트)
    csrf = request.form.get("csrf_token")
    if request.method == "POST" and csrf and csrf == session.get("csrf_token"):
        # 폼 데이터 가져오기
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        topics_str = request.form.get("topics", "").strip()

        # 토픽을 쉼표로 분리하여 리스트로 변환
        topics = [topic.strip() for topic in topics_str.split(",") if topic.strip()]

        # 이메일 형식 확인
        if not is_valid_email(email):
            flash("유효하지 않은 이메일 형식입니다.", "error")
            return render_template(
                "subscriber_form.html", mode="edit", subscriber=subscriber
            )

        # 구독자 정보 업데이트
        for s in subscribers:
            if s.get("id") == subscriber_id:
                s["name"] = name
                s["email"] = email
                s["topics"] = topics
                s["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                break

        # 변경사항 저장
        write_subscribers(subscribers)

        # 로그 기록
        log_to_file(f"구독자 정보 수정: {name} ({email})")

        flash("구독자 정보가 성공적으로 업데이트되었습니다.", "success")
        return redirect(url_for("list_subscribers"))

    # GET 요청 시 폼에 기존 정보 표시
    return render_template("subscriber_form.html", mode="edit", subscriber=subscriber)


@app.route("/subscribers/delete/<subscriber_id>")
def delete_subscriber(subscriber_id):
    # 로그인 확인
    if "logged_in" not in session:
        return redirect(url_for("login"))

    # 구독자 목록 가져오기
    subscribers = load_subscribers()

    # 해당 ID의 구독자 찾기
    subscriber_to_delete = None
    for subscriber in subscribers:
        if subscriber.get("id") == subscriber_id:
            subscriber_to_delete = subscriber
            break

    # 구독자가 존재하지 않는 경우
    if not subscriber_to_delete:
        flash("구독자를 찾을 수 없습니다.", "error")
        return redirect(url_for("list_subscribers"))

    # 구독자 삭제
    subscribers.remove(subscriber_to_delete)
    write_subscribers(subscribers)

    # 로그 기록
    log_to_file(
        f"구독자 삭제: {subscriber_to_delete.get('name')} ({subscriber_to_delete.get('email')})"
    )

    flash("구독자가 성공적으로 삭제되었습니다.", "success")
    return redirect(url_for("list_subscribers"))


if __name__ == "__main__":
    # gunicorn 등으로 구동 시 app:app를 사용
    app.run(host="127.0.0.1", port=5000)
