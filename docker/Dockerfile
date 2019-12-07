FROM python:3.7-slim-buster

RUN apt-get update
RUN apt-get -y install screen

WORKDIR /home

COPY . generative-nids
RUN pip install -r generative-nids/requirements.txt
RUN pip install -e generative-nids