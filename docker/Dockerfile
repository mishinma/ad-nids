FROM python:3.7-buster

ARG ssh_prv_key
ARG ssh_pub_key


RUN apt-get update
RUN apt-get -y install screen zip git python3-dev nano lsof
RUN pip install awscli


# Authorize SSH Host
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh && \
    ssh-keyscan github.com > /root/.ssh/known_hosts

# Add the keys and set permissions
RUN echo "$ssh_prv_key" > /root/.ssh/id_rsa && \
    echo "$ssh_pub_key" > /root/.ssh/id_rsa.pub && \
    chmod 600 /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa.pub

WORKDIR /home

RUN git clone git@github.com:mishinma/alibi-detect.git
RUN pip install -e alibi-detect

COPY . ad-nids
RUN pip install -r ad-nids/requirements/requirements.txt
RUN pip install -r ad-nids/requirements/requirements_notebooks.txt
RUN pip install -e ad-nids

# Remove SSH keys
RUN rm -rf /root/.ssh/