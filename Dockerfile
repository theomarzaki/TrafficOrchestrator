FROM debian:buster

RUN apt-get update

RUN apt-get install -y build-essential \
							g++ \
							netcat \
							wget \
							nodejs \
							cmake \
							unzip \
							vim

COPY . /communication_layer

EXPOSE 7000

WORKDIR /communication_layer

RUN chmod u+x runner.sh