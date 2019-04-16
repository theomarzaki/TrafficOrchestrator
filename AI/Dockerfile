FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN apt update && apt install -y wget curl git gcc build-essential software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt install -y python3.7 python3-dev python3-pip

RUN ln -sf /usr/bin/python3.7 /usr/bin/python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /root

COPY requirements.txt .

RUN python3 --version

RUN pip3 install --upgrade pip
RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp37-cp37m-linux_x86_64.whl
RUN pip3 install -r requirements.txt

RUN mkdir DQN_Saves

COPY ./*.py ./
ADD ./csv/ ./csv

RUN ls

ENTRYPOINT ["python","Model_Runner.py"]

CMD ["--double"]