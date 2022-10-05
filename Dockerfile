FROM python:3.8

COPY . /app

WORKDIR /app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install cmake
RUN pip install -r requirements.txt


CMD "python3" "loadModel.py"

