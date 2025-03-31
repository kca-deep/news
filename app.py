#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import logging
import secrets
import bcrypt
from datetime import datetime, timedelta
from functools import wraps
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    jsonify,
)
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, PasswordField, TextAreaField
from wtforms.validators import DataRequired, Email, Length
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


# 구성 클래스: 환경변수에서 설정값을 중앙집중화
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    NEWS_TOPICS = os.getenv("NEWS_TOPICS", "")
    GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")
    USD_TO_KRW = float(os.getenv("USD_TO_KRW", "1350"))
    EMAIL_USERNAME = os.getenv("EMAIL_USERNAME", "")
    EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
    EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
    GMAIL_CREDENTIALS_FILE = os.getenv("GMAIL_CREDENTIALS_FILE", "credentials.json")
    EMAIL_OAUTH2_TOKEN = os.getenv("EMAIL_OAUTH2_TOKEN", "token.json")
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(16))
    CSRF_SECRET_KEY = os.getenv("CSRF_SECRET_KEY", secrets.token_hex(16))
    DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t", "yes")
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "kca")
    ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "")


# JSON 파일 경로 상수
SUBSCRIBERS_FILE = "subscribers.json"

# 로그 디렉토리 생성 및 로그 설정
if not os.path.exists("logs"):
    os.makedirs("logs")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join("logs", f"newsletter.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger("newsletter")


# 폼 클래스 정의
class LoginForm(FlaskForm):
    """로그인 폼"""

    username = StringField("사용자 이름", validators=[DataRequired()])
    password = PasswordField("비밀번호", validators=[DataRequired()])


class SubscriberForm(FlaskForm):
    """구독자 추가/수정 폼"""

    name = StringField("이름", validators=[DataRequired(), Length(min=1, max=100)])
    email = StringField("이메일", validators=[DataRequired(), Email()])
    topics = TextAreaField("관심 토픽")


# 구독자 클래스 정의
class Subscriber:
    def __init__(self, id, name, email, topics=None, created_at=None):
        self.id = id
        self.name = name
        self.email = email
        self.topics = topics or []
        self.created_at = created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 구독자 관리 함수들
def load_subscribers():
    """subscribers.json 파일에서 구독자 정보를 로드하는 함수"""
    try:
        if not os.path.exists(SUBSCRIBERS_FILE):
            with open(SUBSCRIBERS_FILE, "w", encoding="utf-8") as f:
                f.write("[]")
            return []
        with open(SUBSCRIBERS_FILE, "r", encoding="utf-8") as f:
            subscribers_data = json.load(f)
        subscribers = []
        for s in subscribers_data:
            subscribers.append(
                Subscriber(
                    id=s.get("id"),
                    name=s.get("name"),
                    email=s.get("email"),
                    topics=s.get("topics", []),
                    created_at=s.get("created_at"),
                )
            )
        logger.info(f"구독자 {len(subscribers)}명 로드 완료")
        return subscribers
    except Exception as e:
        logger.error(f"구독자 로드 중 오류 발생: {e}")
        return []


def save_subscribers(subscribers):
    """구독자 정보를 파일에 저장하는 함수"""
    try:
        subscribers_data = []
        for s in subscribers:
            subscribers_data.append(
                {
                    "id": s.id,
                    "name": s.name,
                    "email": s.email,
                    "topics": s.topics,
                    "created_at": s.created_at,
                }
            )
        with open(SUBSCRIBERS_FILE, "w", encoding="utf-8") as f:
            json.dump(subscribers_data, f, ensure_ascii=False, indent=2)
        logger.info(f"구독자 {len(subscribers)}명 저장 완료")
    except Exception as e:
        logger.error(f"구독자 저장 중 오류 발생: {e}")


def get_next_id(subscribers):
    """다음 구독자 ID를 생성하는 함수"""
    if not subscribers:
        return 1
    return max(s.id for s in subscribers) + 1


def find_subscriber_by_id(subscribers, id):
    """ID로 구독자를 찾는 함수"""
    for subscriber in subscribers:
        if subscriber.id == id:
            return subscriber
    return None


def find_subscriber_by_email(subscribers, email):
    """이메일로 구독자를 찾는 함수"""
    for subscriber in subscribers:
        if subscriber.email.lower() == email.lower():
            return subscriber
    return None


def delete_subscriber_by_id(id):
    """ID로 구독자를 삭제하는 함수"""
    try:
        subscribers = load_subscribers()
        subscriber = find_subscriber_by_id(subscribers, id)
        if subscriber:
            subscribers = [s for s in subscribers if s.id != id]
            save_subscribers(subscribers)
            logger.info(f"구독자 '{subscriber.email}' 삭제 완료")
            return True, f"구독자 '{subscriber.name}'이(가) 삭제되었습니다."
        logger.warning(f"삭제 실패: ID {id}인 구독자를 찾을 수 없음")
        return False, "해당 구독자를 찾을 수 없습니다."
    except Exception as e:
        logger.error(f"구독자 삭제 중 오류 발생: {e}")
        return False, "구독자 삭제 중 오류가 발생했습니다."


def update_subscriber_by_id(id, name, email, topics):
    """구독자 정보를 업데이트하는 함수"""
    try:
        subscribers = load_subscribers()
        subscriber = find_subscriber_by_id(subscribers, id)
        if not subscriber:
            logger.warning(f"업데이트 실패: ID {id}인 구독자를 찾을 수 없음")
            return False, "해당 구독자를 찾을 수 없습니다."
        if subscriber.email.lower() != email.lower():
            existing = find_subscriber_by_email(subscribers, email)
            if existing:
                logger.warning(f"업데이트 실패: 이메일 '{email}'은 이미 사용 중")
                return False, f"이메일 '{email}'은 이미 다른 구독자가 사용 중입니다."
        subscriber.name = name
        subscriber.email = email
        subscriber.topics = topics
        save_subscribers(subscribers)
        logger.info(f"구독자 '{email}' 정보 업데이트 완료")
        return True, f"구독자 '{name}'의 정보가 업데이트되었습니다."
    except Exception as e:
        logger.error(f"구독자 업데이트 중 오류 발생: {e}")
        return False, "구독자 정보 업데이트 중 오류가 발생했습니다."


def is_valid_email(email):
    """이메일 형식이 유효한지 확인하는 함수"""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, email) is not None


def hash_password(password):
    """비밀번호를 해시하는 함수"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def check_password(plain_password, hashed_password):
    """해시된 비밀번호와 일치하는지 확인하는 함수"""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


def load_admin_credentials():
    """관리자 계정 정보를 .env 파일에서 직접 로드하는 함수"""
    try:
        username = os.getenv("ADMIN_USERNAME", "kca")
        hashed_password = os.getenv("ADMIN_PASSWORD_HASH")
        if not hashed_password:
            raw_password = os.getenv("ADMIN_PASSWORD", "admin123")
            hashed_password = hash_password(raw_password)
            logger.info("기본 관리자 비밀번호를 해시하여 사용합니다.")
        return username, hashed_password
    except Exception as e:
        logger.error(f"관리자 계정 정보 로드 중 오류 발생: {e}")
        return "kca", hash_password("admin123")


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "logged_in" not in session:
            flash("로그인이 필요합니다.", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated_function


# Flask 앱 설정
app = Flask(__name__)
app.secret_key = Config.SECRET_KEY

csrf = CSRFProtect(app)
app.config["WTF_CSRF_ENABLED"] = True
app.config["WTF_CSRF_SECRET_KEY"] = Config.CSRF_SECRET_KEY
app.config["WTF_CSRF_TIME_LIMIT"] = 3600


@app.errorhandler(400)
def handle_csrf_error(e):
    logger.error("CSRF 토큰 오류 발생")
    return (
        render_template(
            "error.html", error="보안 토큰이 만료되었습니다. 다시 시도해주세요."
        ),
        400,
    )


@app.before_request
def make_session_permanent():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(hours=1)


@app.route("/")
def home():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if "logged_in" in session:
        return redirect(url_for("list_subscribers"))
    form = LoginForm()
    if form.validate_on_submit():
        try:
            username = form.username.data
            password = form.password.data
            admin_username, admin_password_hash = load_admin_credentials()
            if username == admin_username and check_password(
                password, admin_password_hash
            ):
                session["logged_in"] = True
                session["username"] = username
                logger.info(f"관리자 '{username}' 로그인 성공")
                flash("로그인에 성공했습니다.", "success")
                return redirect(url_for("list_subscribers"))
            else:
                logger.warning(
                    f"로그인 실패: 사용자 '{username}'의 인증 정보가 올바르지 않음"
                )
                flash("사용자 이름 또는 비밀번호가 잘못되었습니다.", "error")
        except Exception as e:
            logger.error(f"로그인 처리 중 오류 발생: {e}")
            flash("로그인 처리 중 오류가 발생했습니다. 다시 시도해주세요.", "error")
    return render_template("login.html", form=form)


@app.route("/logout")
def logout():
    username = session.get("username", "Unknown")
    session.clear()
    logger.info(f"관리자 '{username}' 로그아웃")
    flash("로그아웃되었습니다.", "success")
    return redirect(url_for("login"))


@app.route("/subscribers")
@login_required
def list_subscribers():
    subscribers = load_subscribers()
    form = FlaskForm()  # CSRF 토큰을 위한 빈 폼
    return render_template("list_subscribers.html", subscribers=subscribers, form=form)


@app.route("/subscribers/add", methods=["GET", "POST"])
@login_required
def add_subscriber():
    form = SubscriberForm()
    if form.validate_on_submit():
        name = form.name.data.strip()
        email = form.email.data.strip()
        topics_text = form.topics.data.strip()
        topics = [t.strip() for t in topics_text.split(",") if t.strip()]
        subscribers = load_subscribers()
        if find_subscriber_by_email(subscribers, email):
            flash(f"이메일 '{email}'은 이미 등록되어 있습니다.", "error")
            return render_template("subscriber_form.html", mode="add", form=form)
        new_id = get_next_id(subscribers)
        new_subscriber = Subscriber(id=new_id, name=name, email=email, topics=topics)
        subscribers.append(new_subscriber)
        save_subscribers(subscribers)
        flash(f"구독자 '{name}'이(가) 추가되었습니다.", "success")
        return redirect(url_for("list_subscribers"))
    return render_template("subscriber_form.html", mode="add", form=form)


@app.route("/subscribers/edit/<int:id>", methods=["GET", "POST"])
@login_required
def edit_subscriber(id):
    subscribers = load_subscribers()
    subscriber = find_subscriber_by_id(subscribers, id)
    if not subscriber:
        flash("해당 구독자를 찾을 수 없습니다.", "error")
        return redirect(url_for("list_subscribers"))
    form = SubscriberForm()
    if request.method == "GET":
        form.name.data = subscriber.name
        form.email.data = subscriber.email
        form.topics.data = ",".join(subscriber.topics)
    if form.validate_on_submit():
        name = form.name.data.strip()
        email = form.email.data.strip()
        topics_text = form.topics.data.strip()
        topics = [t.strip() for t in topics_text.split(",") if t.strip()]
        if email.lower() != subscriber.email.lower():
            existing = find_subscriber_by_email(subscribers, email)
            if existing:
                flash(f"이메일 '{email}'은 이미 다른 구독자가 사용 중입니다.", "error")
                return render_template(
                    "subscriber_form.html", mode="edit", id=id, form=form
                )
        success, message = update_subscriber_by_id(id, name, email, topics)
        flash(message, "success" if success else "error")
        if success:
            return redirect(url_for("list_subscribers"))
        else:
            return render_template(
                "subscriber_form.html", mode="edit", id=id, form=form
            )
    return render_template(
        "subscriber_form.html",
        mode="edit",
        id=subscriber.id,
        form=form,
        name=subscriber.name,
        email=subscriber.email,
    )


@app.route("/subscribers/delete/<int:id>", methods=["POST"])
@login_required
def delete_subscriber(id):
    form = FlaskForm()  # CSRF 토큰을 위한 빈 폼
    if form.validate_on_submit():
        success, message = delete_subscriber_by_id(id)
        flash(message, "success" if success else "error")
    else:
        flash("CSRF 토큰이 유효하지 않습니다. 다시 시도해주세요.", "error")
    return redirect(url_for("list_subscribers"))


@app.route("/subscribers/search")
@login_required
def search_subscribers():
    query = request.args.get("q", "").strip().lower()
    if not query:
        return redirect(url_for("list_subscribers"))
    subscribers = load_subscribers()
    filtered = []
    for subscriber in subscribers:
        if (
            query in subscriber.name.lower()
            or query in subscriber.email.lower()
            or any(query in topic.lower() for topic in subscriber.topics)
        ):
            filtered.append(subscriber)
    form = FlaskForm()  # CSRF 토큰을 위한 빈 폼
    return render_template(
        "list_subscribers.html", subscribers=filtered, search_query=query, form=form
    )


if __name__ == "__main__":
    logger.info("뉴스레터 관리 시스템 실행 시작")
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
