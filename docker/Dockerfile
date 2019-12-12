FROM python:3.7-slim-buster

RUN apt-get update
RUN apt-get -y install screen
RUN apt-get -y install zip

WORKDIR /home

COPY . ad-nids
RUN pip install -r ad-nids/requirements.txt
RUN pip install -e ad-nids