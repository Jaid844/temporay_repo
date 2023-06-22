FROM ubuntu:latest

RUN apt-get update && apt-get install -y python3 python3-pip

# Install PyTorch CPU version
RUN pip3 install torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /main

COPY . /main

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

CMD ["python3", "main.py"]

