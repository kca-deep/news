import bcrypt

# 비밀번호 해싱
password = "wlsgmddnjs1!"
password_bytes = password.encode("utf-8")
salt = bcrypt.gensalt()
hashed = bcrypt.hashpw(password_bytes, salt)
print(hashed.decode("utf-8"))
