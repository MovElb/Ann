FROM python:3.7

ENV TZ Europe/Moscow

ADD requirements.txt /qaweb/
RUN pip install -U pip -r /qaweb/requirements.txt
RUN python -m spacy download en

ADD setup.py /qaweb
ADD ./qaweb /qaweb/qaweb/
ADD ./configs /qaweb/configs/
ADD meta.msgpack /qaweb/squad2_preprocessed/

RUN pip install -e /qaweb

EXPOSE 8080

WORKDIR /qaweb
