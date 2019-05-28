FROM python:3.7

ENV TZ Europe/Moscow

ADD requirements.txt /qaweb/
RUN pip install -U pip -r /qaweb/requirements.txt
RUN python -m spacy download en

ADD setup.py /qaweb
ADD ./qaweb /qaweb/qaweb/
ADD ./configs /qaweb/configs/

RUN mkdir -p /qaweb/squad2_preprocessed \
    && curl -SL https://storage.googleapis.com/seann/meta.msgpack -o /qaweb/squad2_preprocessed/meta.msgpack

RUN pip install -e /qaweb

WORKDIR /qaweb
