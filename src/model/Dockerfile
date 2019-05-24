FROM nvidia/cuda:10.1-base-ubuntu18.04

#RUN apt update \
#    && apt-get install -y software-properties-common \
#    && add-apt-repository -y ppa:deadsnakes/ppa \
#    && apt install -y python3.7 \
#    && apt install -y python3-pip \
#    && apt-get install -y python3-dev \
#    && python3.7 -m pip install pip

RUN apt-get update \
    && apt-get install -y python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get -y install curl

ADD requirements.txt /bertynet/
RUN pip3 install -U pip -r /bertynet/requirements.txt

ADD setup.py /bertynet
ADD ./bertynet /bertynet/bertynet/
ADD ./configs /bertynet/configs/

RUN pip3 install -e /bertynet

RUN mkdir -p /bertynet/data \
    && curl -SL https://storage.googleapis.com/seann/best_model.pt -o /bertynet/data/best_model.pt \
    && curl -SL https://storage.googleapis.com/seann/meta.msgpack -o /bertynet/data/meta.msgpack

EXPOSE 8081
EXPOSE 8082

WORKDIR /bertynet
