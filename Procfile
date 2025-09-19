web: gunicorn stylemate.wsgi --log-file -
worker: celery -A stylemate worker --loglevel=info
beat: celery -A stylemate beat --loglevel=info