#!/bin/bash

xhost +local:

cd docker

docker compose run -d -u $(id -u $USER) metaheuristics_optimization
