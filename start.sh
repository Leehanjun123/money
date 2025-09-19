#!/bin/bash

# 데이터베이스 마이그레이션 실행
python manage.py migrate --noinput

# 정적 파일 수집
python manage.py collectstatic --noinput

# 슈퍼유저 생성 (선택사항)
# python manage.py createsuperuser --noinput --username admin --email admin@stylemate.com || true

# Gunicorn 서버 시작
PORT=${PORT:-8000}
exec gunicorn stylemate.wsgi --log-file - --bind 0.0.0.0:$PORT