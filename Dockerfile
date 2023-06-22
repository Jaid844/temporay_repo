FROM ubuntu:latest

RUN apt-get update && apt-get install -y python3 python3-pip

# Install PyTorch CPU version

WORKDIR /main

COPY . /main

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

CMD ["python3", "main.py"]

