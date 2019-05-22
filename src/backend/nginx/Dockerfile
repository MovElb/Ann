FROM nginx:latest

ARG CERT_PATH=/etc/ssl/seann.ru

ADD nginx.conf /etc/nginx/nginx.conf
ADD ssl.conf /etc/nginx/ssl.conf


ADD seann.ru.crt $CERT_PATH/seann.ru.crt
ADD seann.ru.key $CERT_PATH/seann.ru.key

RUN chmod +rwxrwxrwx $CERT_PATH/seann.ru.crt
RUN chmod +rwxrwxrwx $CERT_PATH/seann.ru.key
