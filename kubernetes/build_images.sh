#!/bin/bash

docker build -t olivierrenouard1103523/docker-api_predict:latest -f ./../src/Dockerfile.predict ./../src
docker push olivierrenouard1103523/docker-api_predict:latest

docker build -t olivierrenouard1103523/docker-api_oauth:latest -f ./../src/fastapi_oauth/Dockerfile.oauth ./../src/fastapi_oauth
docker push olivierrenouard1103523/docker-api_oauth:latest

docker build -t olivierrenouard1103523/docker-api_train:latest -f ./../src/Dockerfile.train ./../src
docker push olivierrenouard1103523/docker-api_train:latest

docker build -t olivierrenouard1103523/docker-api_flows:latest -f ./../src/flows/Dockerfile.flows ./../src/flows
docker push olivierrenouard1103523/docker-api_flows:latest

docker build -t olivierrenouard1103523/docker-streamlit:latest -f ./../streamlit_app/Dockerfile.streamlit ./../streamlit_app
docker push olivierrenouard1103523/docker-streamlit:latest

docker build -t olivierrenouard1103523/docker-tensorboard:latest -f ./../tensorboard/Dockerfile.tensorboard ./../tensorboard
docker push olivierrenouard1103523/docker-tensorboard:latest
