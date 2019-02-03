FROM ubuntu:latest

RUN apt-get update -y

RUN apt-get install g++ -y

RUN apt install build-essential -y

RUN apt install netcat -y

RUN apt install wget -y

RUN apt install cmake -y

RUN apt install unzip -y

RUN apt install vim -y

COPY . /communication_layer

EXPOSE 7000

WORKDIR /communication_layer

RUN chmod u+x runner.sh

ENTRYPOINT "./runner.sh"
