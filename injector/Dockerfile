FROM ubuntu:latest

RUN apt-get update

RUN apt install build-essential -y

COPY . /to

WORKDIR /to

CMD ["./injector/5gcars-injector-linux"]
