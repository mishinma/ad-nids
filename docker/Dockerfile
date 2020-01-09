FROM python:3.7-buster

RUN apt-get update
RUN apt-get -y install screen zip git python3-dev nano
RUN pip install awscli

RUN pip install --no-cache-dir git+git://github.com/mishinma/alibi-detect.git

WORKDIR /home

COPY . ad-nids
RUN pip install -r ad-nids/requirements/requirements.txt
RUN pip install -r ad-nids/requirements/requirements_notebooks.txt
RUN pip install -e ad-nids
