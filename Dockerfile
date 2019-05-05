FROM python:3.7

ENV TZ Europe/Moscow

ADD requirements.txt /qaweb/
RUN pip install -U pip -r /qaweb/requirements.txt

RUN pip install -U git+https://github.com/Supervisor/supervisor \
    git+https://github.com/MagicStack/uvloop

ADD src/backend /qaweb/backend/
ADD src/configs /qaweb/configs/

RUN pip install -e /qaweb

EXPOSE 8080

WORKDIR /qaweb
