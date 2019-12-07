Are deep generative models suitable for use in anomaly-based network intrusion detection systems?

Docker 
------
::

    docker build -f docker/Dockerfile .  -t generative-nids
    docker run -di --name worker generative-nids
 
