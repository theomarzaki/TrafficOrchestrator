FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN apt update && apt install -y wget curl git gcc build-essential python3 python3-dev python3-pip

WORKDIR /root

COPY requirements.txt .

RUN python3 --version

RUN pip3 install --upgrade pip
RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp35-cp35m-linux_x86_64.whl
RUN pip3 install -r requirements.txt

RUN ls
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN mkdir DQN_Saves

COPY ./*.py ./
ADD ./csv/ ./csv

CMD ["python","Double_DQN.py"]