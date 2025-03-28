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
                f.write(f"토픽: {subscriber['topics']}\n")
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
                            subscribers.append(current_subscriber)
                        else:
                            needs_cleanup = True
                    current_subscriber = {
                        "id": line[3:].strip(),
                        "name": "",
                        "email": "",
                        "topics": "",
                    }
                elif line.startswith("이름:"):
                    current_subscriber["name"] = line[3:].strip()
                elif line.startswith("이메일:"):
                    current_subscriber["email"] = line[4:].strip()
                elif line.startswith("토픽:"):
                    current_subscriber["topics"] = line[3:].strip()

            # 마지막 구독자 정보 처리
            if current_subscriber:
                if current_subscriber.get("name") and current_subscriber.get("email"):
                    if not current_subscriber.get("id"):
                        current_subscriber["id"] = str(uuid.uuid4())
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


# --- Flask 웹 인터페이스 ---
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))
csrf = CSRFProtect(app)

# 관리자 비밀번호 설정 (해시된 값 사용)
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH")
if not ADMIN_PASSWORD_HASH:
    ADMIN_PASSWORD_HASH = hash_password(os.getenv("ADMIN_PASSWORD", "admin123"))
    log_to_file(
        "비밀번호가 해시되어 사용됩니다. .env 파일을 업데이트하는 것을 권장합니다."
    )


@app.before_request
def make_session_permanent():
    """세션을 영구적으로 설정하고 30분 후 만료하도록 구성."""
    session.permanent = True
    app.permanent_session_lifetime = datetime.timedelta(minutes=30)


@app.route("/")
def index():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        expected_username = os.getenv("ADMIN_USERNAME", "kca")
        if username != expected_username:
            flash("로그인 정보가 올바르지 않습니다.")
            log_to_file(f"로그인 실패: 잘못된 사용자 ID - {username}")
            return render_template("login.html")

        if check_password(password, ADMIN_PASSWORD_HASH):
            session["logged_in"] = True
            session["username"] = username
            session["login_time"] = datetime.datetime.now().isoformat()
            log_to_file(f"로그인 성공: {username}")
            return redirect(url_for("list_subscriber_web"))
        else:
            flash("로그인 정보가 올바르지 않습니다.")
            log_to_file("로그인 실패: 패스워드 불일치")
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
        elif not is_valid_email(email):
            flash("유효하지 않은 이메일 형식입니다.")
        else:
            if save_subscriber(name, email, topics):
                flash("구독자 추가에 성공했습니다.")
            else:
                flash("구독자 추가에 실패했습니다. (이미 등록된 이메일일 수 있습니다.)")
    return render_template("add_subscriber.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("로그아웃 되었습니다.")
    return redirect(url_for("login"))


@app.route("/list_subscribers")
@login_required
def list_subscriber_web():
    subs = load_subscribers()
    update_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template(
        "list_subscribers.html", subscribers=subs, update_time=update_time
    )


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
    subscriber = next((sub for sub in subscribers if sub["id"] == id), None)

    if not subscriber:
        flash("구독자를 찾을 수 없습니다.")
        return redirect(url_for("list_subscriber_web"))

    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        topics = request.form.get("topics")
        if not name or not email or not topics:
            flash("모든 항목을 입력해주세요.")
        elif not is_valid_email(email):
            flash("유효하지 않은 이메일 형식입니다.")
        else:
            success, message = update_subscriber_by_id(id, name, email, topics)
            if success:
                flash("구독자 정보가 성공적으로 수정되었습니다.")
                return redirect(url_for("list_subscriber_web"))
            else:
                flash(message)

    topics_str = subscriber.get("topics", "")
    if isinstance(topics_str, list):
        topics_str = ", ".join(topics_str)

    return render_template(
        "edit_subscriber.html", subscriber=subscriber, topics=topics_str
    )


if __name__ == "__main__":
    # gunicorn 등으로 구동 시 app:app를 사용
    app.run(host="127.0.0.1", port=5000)
