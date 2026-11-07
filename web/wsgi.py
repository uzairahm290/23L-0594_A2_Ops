"""Entry point. In the container gunicorn imports `app` from here.
Locally:  FLASK_DEBUG=0 flask --app web.wsgi run   (debug stays off - see report 4.4)"""
from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
