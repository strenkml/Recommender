FROM ghcr.io/linuxserver/baseimage-kasmvnc:ubuntujammy

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y \
        python3.11 \
        python3-pip \
        python3-venv \
        sqlite3 \
        python3-pyqt6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY main.py main.py

RUN mkdir data

RUN python3 -m venv venv
RUN source venv/bin/activate

RUN pip install -r requirement.txt

CMD python main.py