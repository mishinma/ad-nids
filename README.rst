Anomaly detection in network intrusion detection systems

Docker 
------
::

    docker build -f docker/Dockerfile .  -t ad-nids --build-arg ssh_prv_key="$(cat ~/.ssh/id_rsa)" --build-arg ssh_pub_key="$(cat ~/.ssh/id_rsa.pub)"
    docker run -di --name worker -v /storage/nids/:/data/ -v /home/emikmis/dev/ad-nids/notebooks:/home/notebooks/  -p 8888:8888 ad-nids
    tensorboard --host 0.0.0.0 --port 4000 --logdir {logdir}
    jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root --NotebookApp.token=
    kill $(lsof -t -i:4000)
