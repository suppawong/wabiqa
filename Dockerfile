FROM python:3.6-slim

RUN apt-get clean \
    && apt-get -y update
RUN apt-get -y install nginx \
    && apt-get -y install python3-dev \
    && apt-get -y install git \
    && apt-get -y install build-essential


# ADD ["app/corpus", "/app/corpus"]
# ADD ["app/wikipedia_articles", "/app/wikipedia_articles"]
# ADD ["app/model/drqa", "/app/model/drqa"]

RUN mkdir /app
RUN mkdir /app/data
RUN mkdir /root/pythainlp-data
RUN chmod 777 -R /root


ADD ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

ADD . /app


COPY nginx.conf /etc/nginx
RUN chmod +x ./start.sh
CMD ["./start.sh"]