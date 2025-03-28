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
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify,
)
from flask_wtf.csrf import CSRFProtect
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# --- 로깅 설정 ---
pathlib.Path("logs").mkdir(parents=True, exist_ok=True)
log_file = f"logs/app.log"
logging.basicConfig(
    level=logging.WARNING,  # INFO에서 WARNING으로 변경하여 중요 로그만 기록
    handlers=[
        logging.FileHandler(log_file, mode="a", encoding="utf-8", errors="replace"),
    ],
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def log_to_file(message):
    """중요 메시지만 기록하도록 WARNING 레벨 사용"""
    logging.warning(message)


# --- 구독자 관리 함수 ---
def load_subscribers(file_path="subscribers.txt"):
    """텍스트 파일에서 구독자 정보를 로드하는 함수."""
    subscribers = []
    file_needs_update = False
    needs_cleanup = False

    if not os.path.exists(file_path):
        return subscribers
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            current_subscriber = None
            for line in f:
                line = line.strip()
                if not line:
                    if current_subscriber:
                        # 빈 구독자 정보(이름과 이메일이 없는 경우)는 무시
                        if current_subscriber.get("name") and current_subscriber.get(
                            "email"
                        ):
                            # ID가 없는 경우에만 UUID 생성
                            if (
                                "id" not in current_subscriber
                                or not current_subscriber["id"]
                            ):
                                current_subscriber["id"] = str(uuid.uuid4())
                                file_needs_update = True
                            subscribers.append(current_subscriber)
                        else:
                            # 빈 구독자 정보는 파일 정리 대상
                            needs_cleanup = True
                        current_subscriber = None
                    continue

                if line.startswith("ID:"):
                    if current_subscriber:
                        # 빈 구독자 정보(이름과 이메일이 없는 경우)는 무시
                        if current_subscriber.get("name") and current_subscriber.get(
                            "email"
                        ):
                            # ID가 없는 경우에만 UUID 생성
                            if (
                                "id" not in current_subscriber
                                or not current_subscriber["id"]
                            ):
                                current_subscriber["id"] = str(uuid.uuid4())
                                file_needs_update = True
                            subscribers.append(current_subscriber)
                        else:
                            # 빈 구독자 정보는 파일 정리 대상
                            needs_cleanup = True
                    current_subscriber = {
                        "id": line[3:].strip(),
                        "name": "",
                        "email": "",
                        "topics": "",
                    }
                elif line.startswith("이름:"):
                    # 이름 줄 처리
                    name_value = line[3:].strip()
                    if not current_subscriber:
                        # 이름으로 시작하는 첫 줄인 경우 (이전 형식 지원)
                        current_subscriber = {
                            "id": str(uuid.uuid4()),
                            "name": name_value,
                            "email": "",
                            "topics": "",
                        }
                        file_needs_update = True
                    else:
                        # 이미 구독자 정보가 시작된 경우
                        current_subscriber["name"] = name_value
                elif current_subscriber:
                    if line.startswith("이메일:"):
                        current_subscriber["email"] = line[4:].strip()
                    elif line.startswith("토픽:"):
                        current_subscriber["topics"] = line[3:].strip()

            # 마지막 구독자 정보 처리
            if current_subscriber:
                # 빈 구독자 정보(이름과 이메일이 없는 경우)는 무시
                if current_subscriber.get("name") and current_subscriber.get("email"):
                    if "id" not in current_subscriber or not current_subscriber["id"]:
                        current_subscriber["id"] = str(uuid.uuid4())
                        file_needs_update = True
                    subscribers.append(current_subscriber)
                else:
                    # 빈 구독자 정보는 파일 정리 대상
                    needs_cleanup = True

        # 파일 정리 또는 업데이트 필요 시에만 파일 쓰기
        if (needs_cleanup or file_needs_update) and subscribers:
            with open(file_path, "w", encoding="utf-8", errors="replace") as f:
                for i, subscriber in enumerate(subscribers):
                    if i > 0:
                        f.write("\n")
                    f.write(f"ID: {subscriber['id']}\n")
                    f.write(f"이름: {subscriber['name']}\n")
                    f.write(f"이메일: {subscriber['email']}\n")
                    f.write(f"토픽: {subscriber['topics']}\n")

        return subscribers
    except Exception as e:
        log_to_file(f"구독자 정보 로드 오류: {e}")
        return []


def save_subscriber(name, email, topics, file_path="subscribers.txt"):
    """새 구독자 정보를 파일에 저장하는 함수."""
    try:
        subscribers = load_subscribers(file_path)
        for subscriber in subscribers:
            if subscriber["email"] == email:
                return False

        # UUID 생성
        subscriber_id = str(uuid.uuid4())

        with open(file_path, "a", encoding="utf-8", errors="replace") as f:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                f.write("\n")
            f.write("\n")
            f.write(f"ID: {subscriber_id}\n")
            f.write(f"이름: {name}\n")
            f.write(f"이메일: {email}\n")
            f.write(f"토픽: {topics}\n")
        log_to_file(f"구독자 추가: {name} ({email})")
        return True
    except Exception as e:
        log_to_file(f"구독자 추가 오류: {e}")
        return False


def delete_subscriber_by_id(subscriber_id, file_path="subscribers.txt"):
    """파일에서 구독자 정보를 ID로 삭제하는 함수."""
    try:
        subscribers = load_subscribers(file_path)
        new_subscribers = [
            subscriber
            for subscriber in subscribers
            if subscriber["id"] != subscriber_id
        ]
        if len(new_subscribers) == len(subscribers):
            return False
        with open(file_path, "w", encoding="utf-8", errors="replace") as f:
            for i, subscriber in enumerate(new_subscribers):
                if i > 0:
                    f.write("\n")
                f.write(f"ID: {subscriber['id']}\n")
                f.write(f"이름: {subscriber['name']}\n")
                f.write(f"이메일: {subscriber['email']}\n")
                f.write(f"토픽: {subscriber['topics']}\n")
        log_to_file(f"구독자 삭제: ID {subscriber_id}")
        return True
    except Exception as e:
        log_to_file(f"구독자 삭제 오류: {e}")
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
                    for other in subscribers:
                        if other["id"] != subscriber_id and other["email"] == email:
                            return False, "이미 존재하는 이메일입니다."

                # 구독자 정보 업데이트
                subscriber["name"] = name
                subscriber["email"] = email
                subscriber["topics"] = topics
                updated = True
                break

        if not updated:
            return False, "구독자를 찾을 수 없습니다."

        # 파일에 저장
        with open(file_path, "w", encoding="utf-8", errors="replace") as f:
            for i, subscriber in enumerate(subscribers):
                if i > 0:
                    f.write("\n")
                f.write(f"ID: {subscriber['id']}\n")
                f.write(f"이름: {subscriber['name']}\n")
                f.write(f"이메일: {subscriber['email']}\n")
                f.write(f"토픽: {subscriber['topics']}\n")

        log_to_file(
            f"구독자 정보 수정: ID {subscriber_id}, 이름: {name}, 이메일: {email}"
        )
        return True, ""
    except Exception as e:
        log_to_file(f"구독자 정보 수정 오류: {e}")
        return False, f"수정 중 오류 발생: {e}"


# 이전 버전 지원을 위한 함수
def delete_subscriber(email, file_path="subscribers.txt"):
    """파일에서 구독자 정보를 이메일로 삭제하는 함수 (하위 호환성용)."""
    try:
        subscribers = load_subscribers(file_path)
        # 이메일로 구독자 ID 찾기
        for subscriber in subscribers:
            if subscriber["email"] == email:
                return delete_subscriber_by_id(subscriber["id"], file_path)
        log_to_file(f"삭제할 구독자를 찾을 수 없습니다: {email}")
        return False
    except Exception as e:
        log_to_file(f"구독자 삭제 오류: {e}")
        return False


# --- 비밀번호 해싱 함수 ---
def hash_password(password):
    """비밀번호를 bcrypt로 해싱하는 함수"""
    # 비밀번호를 바이트로 인코딩
    password_bytes = password.encode("utf-8")
    # 솔트 생성 및 해싱
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")  # 문자열로 변환하여 반환


def check_password(plain_password, hashed_password):
    """해시된 비밀번호와 일반 텍스트 비밀번호가 일치하는지 확인하는 함수"""
    # 문자열 비밀번호를 바이트로 변환
    plain_bytes = plain_password.encode("utf-8")
    hashed_bytes = hashed_password.encode("utf-8")
    # 비밀번호 확인
    return bcrypt.checkpw(plain_bytes, hashed_bytes)


# --- 세션 필요 데코레이터 ---
def login_required(func):
    """로그인이 필요한 라우트에 적용할 데코레이터"""

    @functools.wraps(func)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            flash("로그인이 필요한 페이지입니다.")
            return redirect(url_for("login"))
        return func(*args, **kwargs)

    return decorated_function


# --- Flask 웹 인터페이스 ---
app = Flask(__name__)
# 강력한 secret_key 사용
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))
# CSRF 보호 설정
csrf = CSRFProtect(app)
# 해시된 비밀번호 사용
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH")

# 해시된 비밀번호가 없는 경우 .env의 일반 텍스트 비밀번호를 해싱하여 사용
if not ADMIN_PASSWORD_HASH:
    ADMIN_PASSWORD_HASH = hash_password(os.getenv("ADMIN_PASSWORD", "admin123"))
    log_to_file(
        "비밀번호가 해시되어 사용됩니다. .env 파일을 업데이트하는 것을 권장합니다."
    )


# 세션 설정 향상
@app.before_request
def make_session_permanent():
    """세션 설정을 향상시키는 함수"""
    session.permanent = True  # 영구 세션 활성화
    app.permanent_session_lifetime = 1800  # 세션 유효기간 30분으로 설정


@app.route("/")
def index():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # 사용자 ID가 kca인지 확인
        expected_username = os.getenv("ADMIN_USERNAME", "kca")
        if username != expected_username:
            flash("로그인 정보가 올바르지 않습니다.")
            log_to_file(f"로그인 실패: 잘못된 사용자 ID - {username}")
            return render_template("login.html")

        # 해시된 비밀번호와 비교
        if check_password(password, ADMIN_PASSWORD_HASH):
            session["logged_in"] = True
            session["username"] = username
            session["login_time"] = secrets.token_hex(16)  # 세션 고유성 부여
            log_to_file(f"로그인 성공: {username}")
            return redirect(url_for("list_subscriber_web"))
        else:
            flash("로그인 정보가 올바르지 않습니다.")
            log_to_file("로그인 실패: 패스워드 불일치")
    # CSRF 토큰을 템플릿에 제공
    return render_template("login.html")


@app.route("/add_subscriber_web", methods=["GET", "POST"])
@login_required
def add_subscriber_web():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        topics = request.form.get("topics")
        if not name or not email or not topics:
            flash("모든 항목을 입력해주세요.")
        else:
            if save_subscriber(name, email, topics):
                flash("구독자 추가에 성공했습니다.")
            else:
                flash("구독자 추가에 실패했습니다. (이미 등록된 이메일일 수 있습니다.)")
    return render_template("add_subscriber.html")


@app.route("/logout")
def logout():
    session.clear()  # 모든 세션 데이터 삭제
    flash("로그아웃 되었습니다.")
    return redirect(url_for("login"))


@app.route("/list_subscribers")
@login_required
def list_subscriber_web():
    subs = load_subscribers()
    # 현재 시간 포맷
    update_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template(
        "list_subscribers.html", subscribers=subs, update_time=update_time
    )


# 구독자 삭제 라우트 (ID 기반)
@app.route("/delete_subscriber/<id>", methods=["GET", "POST"])
@login_required
def delete_subscriber_web(id):
    if request.method == "POST":
        if delete_subscriber_by_id(id):
            flash("구독자가 성공적으로 삭제되었습니다.")
        else:
            flash("구독자 삭제에 실패했습니다.")
    else:
        flash("삭제는 POST 요청으로만 가능합니다.")

    return redirect(url_for("list_subscriber_web"))


@app.route("/edit_subscriber/<id>", methods=["GET", "POST"])
@login_required
def edit_subscriber_web(id):
    subscribers = load_subscribers()
    subscriber = None

    # 구독자 찾기
    for sub in subscribers:
        if sub["id"] == id:
            subscriber = sub
            break

    if not subscriber:
        flash("구독자를 찾을 수 없습니다.")
        return redirect(url_for("list_subscriber_web"))

    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        topics = request.form.get("topics")

        if not name or not email or not topics:
            flash("모든 항목을 입력해주세요.")
        else:
            success, message = update_subscriber_by_id(id, name, email, topics)
            if success:
                flash("구독자 정보가 성공적으로 수정되었습니다.")
                return redirect(url_for("list_subscriber_web"))
            else:
                flash(message)

    # 토픽 목록을 문자열로 변환
    topics_str = subscriber.get("topics", "")
    if isinstance(topics_str, list):
        topics_str = ", ".join(topics_str)

    return render_template(
        "edit_subscriber.html", subscriber=subscriber, topics=topics_str
    )


if __name__ == "__main__":
    # app:app는 app.py 파일의 Flask 인스턴스(app)를 의미
    # gunicorn -w 4 -b 127.0.0.1:5000 app:app 로 구동
    app.run(host="127.0.0.1", port=5000)
