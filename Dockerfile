FROM ghcr.io/linuxserver/baseimage-kasmvnc:ubuntujammy

ENV TITLE="Recommender"
ENV CUSTOM_USER="recommender"
ENV FM_HOME="/app"
ENV START_DOCKER=false


WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/app/venv \
    PATH="/app/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y \
        python3.11 \
        python3-pip \
        python3-venv \
        sqlite3 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN python3 -m venv $VIRTUAL_ENV \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt pyqt6

COPY main.py main.py

RUN mkdir -p /app/data && chown -R 1000:1000 /app/data

RUN mkdir -p /root/defaults
RUN echo "cd /app && source venv/bin/activate && /usr/bin/python3 main.py" > /root/defaults/autostart

CMD ["tail", "-f", "/dev/null"]