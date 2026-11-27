"""Entry point for gunicorn, the production web server used in the Docker image.

gunicorn needs a module-level variable that holds the Flask app, so we
build it here once when the server starts.

Run (Linux / Docker only - gunicorn does not run on Windows):
    gunicorn --bind 0.0.0.0:8000 src.api.wsgi:app
On Windows use the development server instead: python -m src.api.app
"""

from src.api.app import create_app, load_artifacts

app = create_app(load_artifacts())
