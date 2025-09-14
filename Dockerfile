# syntax=docker/dockerfile:1

FROM python:3.10-slim 

# WORKDIR DOCKER
WORKDIR /app

# install depsies n run it
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy script
COPY iris_classification.py iris.csv data_uji.csv ./

# default command
CMD [ "python", "iris_classification.py", "--help" ]
