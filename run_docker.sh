#!/bin/bash

docker build --file Dockerfile --tag kws-pytorch .
docker run -it --gpus all --ipc=host -p 8080:8080 -v /home/$USER/KWS/:/home/$USER kws-pytorch:latest bash