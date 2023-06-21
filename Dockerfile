FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y python3 python3-pip

WORKDIR /app

COPY . /app

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

CMD ["python3", "app.py"]
