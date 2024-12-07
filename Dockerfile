# Leaving Dockerfile here for reference but does not make sense to use it for now because samples are stored in local disk.
FROM python:3.11-slim

WORKDIR /app

ARG APP_PORT=8501

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=42 \
  # Poetry's configuration:
  POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_CREATE=false \
  POETRY_HOME="/usr/local" \
  POETRY_VERSION=1.8.3

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -

# Requirements for torch instalation in MacOS. If installing in a different OS or architecture, add a corresponding requirements.txt file.
COPY poetry.lock pyproject.toml requirements.macos.txt ./

RUN poetry install --only=main

RUN pip install --no-cache-dir -r /app/requirements.macos.txt

COPY src /app/src
# Copying only the CLAP weights file that I'm using for the demo to keep the image size smaller.
COPY checkpoints/CLAP_weights_2023.pth /app/checkpoints/

EXPOSE ${APP_PORT}

HEALTHCHECK CMD curl --fail http://localhost:${APP_PORT}/_stcore/health

ENTRYPOINT ["sh", "-c", "streamlit run src/demo.py --server.port=${APP_PORT} --server.address=0.0.0.0"]
