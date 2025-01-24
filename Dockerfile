FROM ghcr.io/linuxserver/baseimage-kasmvnc:debianbookworm

ENV TITLE="Recommender"

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
        python3-pyqt6 \
        libx11-xcb1 \
        libxcb-render0 \
        libxcb-shm0 \
        libxcb-xfixes0 \
        libxcb-shape0 \
        libxcb-randr0 \
        libxcb-image0 \
        libxcb-icccm4 \
        libxcb-keysyms1 \
        libxcb-sync1 \
        libxcb-xinerama0 \
        libxcb-util1 \
        libxkbcommon-x11-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN python3 -m venv $VIRTUAL_ENV \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py main.py

RUN mkdir -p /app/data && chown -R 1000:1000 /app/data

RUN mkdir -p /root/defaults
RUN echo "cd /app && source venv/bin/activate && /usr/bin/python3 main.py" > /root/defaults/autostart
RUN chmod +x /root/defaults/autostart

EXPOSE 3000
