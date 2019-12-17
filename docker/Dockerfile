FROM python:3.7-buster

RUN apt-get update
RUN apt-get -y install screen zip git python3-dev

RUN pip install git+git://github.com/mishinma/alibi-detect.git

WORKDIR /home

COPY . ad-nids
RUN pip install -r ad-nids/requirements.txt
RUN pip install -e ad-nids